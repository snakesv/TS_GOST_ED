"""Certified TS-GOST-ED implementation for EJMCA_LMTU_TS_GOST_ED_v1.tex.

Profile: prime-order Ed25519 subgroup, SHA-256, independent Ed25519 Auth keys.
This is NOT a standard GOST signature implementation. See README.md for limits.
All public protocol objects are JSON-compatible dictionaries. Nonces stay in RAM;
a SQLite reservation burns the session identifier before any commitment is sent.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
from typing import Callable

from Crypto.Hash import SHA256
from Crypto.PublicKey import ECC
from Crypto.Random import get_random_bytes
from Crypto.Signature import eddsa

P = 2**255 - 19
N = 2**252 + 27742317777372353535851937790883648493
D = (-121665 * pow(121666, -1, P)) % P
GX = 15112221349535400772501151409588531511454012693041857206046113283949847762202
GY = 46316835694926478169428394003475163141307993866256225615783033603165251855960
G = ECC.EccPoint(GX, GY, curve="Ed25519")
O = G * 0
SUITE = "TSGOSTED-ED25519-SHA256-EDDSA-LMTU-v1"


class ProtocolError(ValueError):
    """Invalid input, invalid contribution, or forbidden state transition."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def canonical(obj) -> bytes:
    """Deterministic JSON; typed strings/integers and explicit array boundaries."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def frame(tag: str, *values) -> bytes:
    return canonical([SUITE, tag, *values])


def clone(obj):
    return json.loads(canonical(obj))


def digest(tag: str, *values) -> str:
    return SHA256.new(frame(tag, *values)).hexdigest()


def strict_hex(value: str, size: int) -> bytes:
    require(type(value) is str and len(value) == size * 2, "Invalid hex length")
    try:
        data = bytes.fromhex(value)
    except ValueError as exc:
        raise ProtocolError("Invalid hex") from exc
    require(len(data) == size and data.hex() == value, "Non-canonical hex")
    return data


def scalar(value: int) -> str:
    require(type(value) is int and 0 <= value < N, "Scalar outside field")
    return value.to_bytes(32, "big").hex()


def parse_scalar(value: str, nonzero=False) -> int:
    x = int.from_bytes(strict_hex(value, 32), "big")
    require((1 if nonzero else 0) <= x < N, "Invalid scalar range")
    return x


def random_scalar(nonzero=True) -> int:
    while True:
        x = int.from_bytes(get_random_bytes(32), "big")
        if (1 if nonzero else 0) <= x < N:
            return x


def hash_scalar(tag: str, *values) -> int:
    """Rejection sampling into Z_n^*, rather than biased modulo reduction."""
    for counter in range(1 << 32):
        x = int.from_bytes(SHA256.new(frame(tag, *values, counter)).digest(), "big")
        if 0 < x < N:
            return x
    raise ProtocolError("Hash-to-field rejection limit")


def encode_point(point) -> str:
    x, y = int(point.x), int(point.y)
    return (y | ((x & 1) << 255)).to_bytes(32, "little").hex()


@lru_cache(maxsize=4096)
def _decode_coordinates(encoded: str):
    data = strict_hex(encoded, 32)
    v = int.from_bytes(data, "little")
    sign, y = v >> 255, v & ((1 << 255) - 1)
    require(y < P, "Non-canonical point coordinate")
    den = (D * y * y + 1) % P
    require(den != 0, "Invalid point denominator")
    xx = ((y*y - 1) * pow(den, -1, P)) % P
    x = pow(xx, (P+3)//8, P)
    if x*x % P != xx:
        x = x * pow(2, (P-1)//4, P) % P
    require(x*x % P == xx, "Point is not on the Edwards curve")
    require(x != 0 or sign == 0, "Non-canonical sign of zero")
    if (x & 1) != sign:
        x = P-x
    point = ECC.EccPoint(x, y, curve="Ed25519")
    # Explicit double-and-add avoids assuming backend scalar reduction semantics.
    acc, base, k = O.copy(), point.copy(), N
    while k:
        if k & 1:
            acc += base
        base += base
        k >>= 1
    require(acc == O, "Point is outside the prime-order subgroup")
    return x, y


def decode_point(encoded: str, nonzero=False):
    point = ECC.EccPoint(*_decode_coordinates(encoded), curve="Ed25519")
    require(not nonzero or point != O, "Identity point is forbidden")
    return point


def keys(obj, expected):
    require(type(obj) is dict and set(obj) == set(expected), "Unexpected object fields")


def auth_public(private_key) -> str:
    return private_key.public_key().export_key(format="raw").hex()


def auth_sign(private_key, tag: str, *values) -> str:
    return eddsa.new(private_key, "rfc8032").sign(frame(tag, *values)).hex()


def auth_check(public_key: str, signature: str, tag: str, *values) -> None:
    decode_point(public_key, nonzero=True)
    sig = strict_hex(signature, 64)
    try:
        eddsa.new(eddsa.import_public_key(strict_hex(public_key, 32)),
                  "rfc8032").verify(frame(tag, *values), sig)
    except (ValueError, TypeError) as exc:
        raise ProtocolError("Invalid authentication signature") from exc


def lagrange(ids: list[int]) -> dict[int, int]:
    require(ids and all(type(i) is int and 0 < i < N for i in ids), "Invalid identifiers")
    require(ids == sorted(set(ids)), "Identifiers must be distinct and ordered")
    out = {}
    for i in ids:
        value = 1
        for j in ids:
            if j != i:
                value = value * (-j) * pow(i-j, -1, N) % N
        out[i] = value
    return out


def cs_keygen():
    """Return public parameters and a trapdoor for isolated decryption tests."""
    w = random_scalar()
    g2 = w*G
    x1,x2,y1,y2,z = [random_scalar() for _ in range(5)]
    pk = dict(g2=encode_point(g2), X=encode_point(x1*G+x2*g2),
              Y=encode_point(y1*G+y2*g2), Z=encode_point(z*G))
    return pk, (x1,x2,y1,y2,z)


def cs_aux(pk, label, ciphertext):
    keys(ciphertext, ("u1", "u2", "E", "V"))
    u1,u2,E,V = [decode_point(ciphertext[k]) for k in ("u1","u2","E","V")]
    require(u1 != O and u2 != O, "Degenerate encryption randomness")
    alpha = hash_scalar("CS-TAG", label, ciphertext["u1"], ciphertext["u2"], ciphertext["E"])
    W = decode_point(pk["X"])+alpha*decode_point(pk["Y"])
    return u1,u2,E,V,W


def cs_encrypt(pk, label, k: int, rho: int):
    require(0 < k < N and 0 < rho < N, "Nonzero encryption witnesses required")
    Q = k*G
    ct = dict(u1=encode_point(rho*G), u2=encode_point(rho*decode_point(pk["g2"])),
              E=encode_point(rho*decode_point(pk["Z"])+Q))
    a = hash_scalar("CS-TAG", label, ct["u1"],ct["u2"],ct["E"])
    ct["V"] = encode_point(rho*(decode_point(pk["X"])+a*decode_point(pk["Y"])))
    return ct


def cs_decrypt(pk, secret, label, ciphertext):
    u1,u2,E,V,_ = cs_aux(pk,label,ciphertext)
    x1,x2,y1,y2,z = secret
    a = hash_scalar("CS-TAG",label,ciphertext["u1"],ciphertext["u2"],ciphertext["E"])
    require(((x1+a*y1)%N)*u1+((x2+a*y2)%N)*u2 == V, "Invalid Cramer-Shoup tag")
    Q = E + -(z*u1)
    require(Q != O, "Identity plaintext")
    return encode_point(Q)


def prove(pk, label, lock_digest, identity, ct, Q, k, rho):
    u1,u2,E,V,W = cs_aux(pk,label,ct)
    require(decode_point(Q,True) == k*G, "Wrong nonce witness")
    require(ct == cs_encrypt(pk,label,k,rho), "Wrong encryption witness")
    a,b = random_scalar(False), random_scalar(False)
    T = [a*G,b*G,b*decode_point(pk["g2"]),b*decode_point(pk["Z"]),b*W]
    e = hash_scalar("RFC8235-CS",pk,label,lock_digest,identity,ct,Q,
                    [encode_point(t) for t in T])
    return dict(e=scalar(e), zk=scalar((a-e*k)%N), zr=scalar((b-e*rho)%N))


def verify_proof(pk,label,lock_digest,identity,ct,Q,proof):
    keys(proof,("e","zk","zr"))
    e,a,b = parse_scalar(proof["e"],True),parse_scalar(proof["zk"]),parse_scalar(proof["zr"])
    point = decode_point(Q,True)
    u1,u2,E,V,W = cs_aux(pk,label,ct)
    T = [a*G+e*point, b*G+e*u1, b*decode_point(pk["g2"])+e*u2,
         b*decode_point(pk["Z"])+e*(E+-point), b*W+e*V]
    require(e == hash_scalar("RFC8235-CS",pk,label,lock_digest,identity,ct,Q,
                             [encode_point(t) for t in T]), "Invalid consistency proof")


class Registry:
    """A CA-authenticated public record; callers pin the CA key independently."""
    def __init__(self, record, ca_signature, pinned_ca_public):
        record = clone(record)
        keys(record,("suite","epoch","threshold","total","P","shares","feldman","cs","auth"))
        auth_check(pinned_ca_public,ca_signature,"REGISTRY",record)
        require(record["suite"] == SUITE, "Wrong ciphersuite")
        t,total = record["threshold"],record["total"]
        require(type(t) is int and type(total) is int and 1 <= t <= total <= 256, "Invalid group size")
        require(type(record["epoch"]) is str and 0 < len(record["epoch"]) <= 128, "Invalid epoch")
        expected = {str(i) for i in range(1,total+1)}
        require(set(record["shares"]) == set(record["auth"]) == expected, "Invalid member registry")
        require(len(record["feldman"]) == t and record["P"] == record["feldman"][0], "Invalid Feldman commitments")
        decode_point(record["P"],True)
        coeff = [decode_point(v) for v in record["feldman"]]
        keys(record["cs"],("g2","X","Y","Z"))
        for key in record["cs"]:
            decode_point(record["cs"][key], key in ("g2","Z"))
        require(len(set(record["auth"].values())) == total, "Duplicate authentication keys")
        for i in range(1,total+1):
            calculated = O.copy()
            for j,A in enumerate(coeff):
                calculated += pow(i,j,N)*A
            require(calculated == decode_point(record["shares"][str(i)]), "Invalid public share")
            decode_point(record["auth"][str(i)],True)
        self._encoded = canonical(record)
        self.ca_signature = ca_signature
        self.ca_public = pinned_ca_public

    @property
    def record(self):
        return json.loads(self._encoded)

    @property
    def fingerprint(self):
        return SHA256.new(self._encoded).hexdigest()

    def export(self):
        return dict(record=self.record, ca_signature=self.ca_signature)


def trusted_setup(total=5, threshold=3, epoch="research-epoch-1"):
    """Simulate an honest dealer/CA; real share delivery needs a secure channel."""
    require(type(total) is int and type(threshold) is int and 1 <= threshold <= total <= 256,
            "Invalid setup parameters")
    ca = ECC.generate(curve="Ed25519")
    auth = {i:ECC.generate(curve="Ed25519") for i in range(1,total+1)}
    polynomial = [random_scalar()] + [random_scalar(False) for _ in range(threshold-1)]
    shares = {i:sum(a*pow(i,j,N) for j,a in enumerate(polynomial))%N for i in auth}
    cs, trapdoor = cs_keygen()
    record = dict(suite=SUITE,epoch=epoch,threshold=threshold,total=total,
                  P=encode_point(polynomial[0]*G),shares={str(i):encode_point(x*G) for i,x in shares.items()},
                  feldman=[encode_point(a*G) for a in polynomial],cs=cs,
                  auth={str(i):auth_public(key) for i,key in auth.items()})
    # Simulate local share receipt and signed setup acknowledgements (Section 5.1).
    # A deployment must deliver each share over a confidential authenticated channel.
    for i,x in shares.items():
        require(encode_point(x*G)==record["shares"][str(i)], "Setup share check failed")
        ack = auth_sign(auth[i],"SETUP-ACK",i,record)
        auth_check(record["auth"][str(i)],ack,"SETUP-ACK",i,record)
    signature = auth_sign(ca,"REGISTRY",record)
    registry = Registry(record,signature,auth_public(ca))
    # Dropping Python references is not secure memory erasure.
    del polynomial,trapdoor,ca
    return registry, {i:(shares[i],auth[i]) for i in auth}


def make_context(registry: Registry, message: bytes, signers: list[int], timestamp=None, sid=None):
    require(type(message) is bytes, "Message must be bytes")
    r = registry.record
    context = dict(registry=registry.fingerprint,epoch=r["epoch"],message=message.hex(),
                   timestamp=int(time.time()) if timestamp is None else timestamp,
                   sid=get_random_bytes(32).hex() if sid is None else sid,signers=list(signers))
    validate_context(registry,context)
    return context


def validate_context(registry,context):
    keys(context,("registry","epoch","message","timestamp","sid","signers"))
    r = registry.record
    require(context["registry"] == registry.fingerprint and context["epoch"] == r["epoch"], "Wrong registry or epoch")
    strict_hex(context["sid"],32)
    require(type(context["timestamp"]) is int and 0 <= context["timestamp"] < 2**63, "Invalid timestamp")
    require(type(context["message"]) is str and len(context["message"]) <= 2*1024*1024, "Invalid message length")
    strict_hex(context["message"],len(context["message"])//2)
    ids = context["signers"]
    require(type(ids) is list and r["threshold"] <= len(ids) <= r["total"], "Invalid signer count")
    lagrange(ids)
    require(all(1 <= i <= r["total"] for i in ids), "Unknown signer")


def label(context,identity):
    return [context,identity]


def eta(context,d):
    return [context["epoch"],context["sid"],context["signers"],d]


def challenge(registry,context,d,Q):
    return hash_scalar("SIGN",Q,registry.record["P"],context["message"],
                       context["timestamp"],eta(context,d))


def approval(registry,context,d,Q):
    return ["TSGOSTED-CERT-v1",registry.record,context["message"],
            context["timestamp"],eta(context,d),Q]


def packet(key,tag,context,identity,data):
    payload = clone(dict(tag=tag,context=context,identity=identity,data=data))
    return dict(payload=payload,signature=auth_sign(key,"TRANSPORT",payload))


def check_packets(registry,context,packets,tag):
    require(type(packets) is list and len(packets)==len(context["signers"]), "Missing or extra packets")
    found = {}
    for pkt in packets:
        keys(pkt,("payload","signature"))
        p = pkt["payload"]
        keys(p,("tag","context","identity","data"))
        i = p["identity"]
        require(type(i) is int and i in context["signers"] and i not in found, "Duplicate or unknown sender")
        require(p["tag"] == tag and p["context"] == context, "Wrong packet context or step")
        auth_check(registry.record["auth"][str(i)],pkt["signature"],"TRANSPORT",p)
        found[i] = clone(p["data"])
    return found


class NonceJournal:
    """Fail-closed session tombstones and cached outputs; no secret nonces on disk.

    One file per signer, owned by the local application. SQLite handles concurrent
    writers. Restoring old backups/deleting the file defeats rollback protection.
    """
    def __init__(self,path,owner):
        require(str(path) != ":memory:", "A durable journal path is required")
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.path,self.owner = str(path),owner
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS owner (id TEXT PRIMARY KEY)")
            rows = db.execute("SELECT id FROM owner").fetchall()
            require(not rows or rows == [(owner,)], "Journal belongs to a different signer")
            db.execute("INSERT OR IGNORE INTO owner VALUES (?)",(owner,))
            db.execute("CREATE TABLE IF NOT EXISTS sessions (epoch TEXT, sid TEXT, context BLOB, state TEXT, digest TEXT, output BLOB, PRIMARY KEY(epoch,sid))")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path,timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def reserve(self,c):
        try:
            with self.connect() as db:
                db.execute("INSERT INTO sessions VALUES (?,?,?,'COMMITTED',NULL,NULL)",
                           (c["epoch"],c["sid"],canonical(c)))
        except sqlite3.IntegrityError as exc:
            raise ProtocolError("Session identifier has already been used") from exc

    def transition(self,c,before,after,d=None,output=None):
        with self.connect() as db:
            cur = db.execute("UPDATE sessions SET state=?, digest=COALESCE(?,digest), output=COALESCE(?,output) WHERE epoch=? AND sid=? AND context=? AND state=?",
                             (after,d,None if output is None else canonical(output),c["epoch"],c["sid"],canonical(c),before))
            require(cur.rowcount == 1,"Journal state transition refused")

    def cached(self,c):
        with self.connect() as db:
            row = db.execute("SELECT output FROM sessions WHERE epoch=? AND sid=? AND context=? AND state='CONSUMED'",
                             (c["epoch"],c["sid"],canonical(c))).fetchone()
        require(row is not None and row[0] is not None,"No completed cached output")
        return json.loads(row[0])

    def abort(self,c):
        with self.connect() as db:
            db.execute("UPDATE sessions SET state='ABORTED' WHERE epoch=? AND sid=? AND context=? AND state!='CONSUMED'",
                       (c["epoch"],c["sid"],canonical(c)))


class Signer:
    def __init__(self,identity,share,auth_key,registry,journal_path,
                 authorize: Callable[[bytes,dict],bool],clock=time.time,max_age=300):
        r = registry.record
        require(type(identity) is int and str(identity) in r["shares"],"Unknown identity")
        require(type(share) is int and 0 <= share < N and encode_point(share*G)==r["shares"][str(identity)],"Wrong secret share")
        require(auth_public(auth_key)==r["auth"][str(identity)] and auth_key.has_private(),"Wrong authentication key")
        require(callable(authorize),"Explicit authorization policy is required")
        self.identity,self.share,self.key,self.registry = identity,share,auth_key,registry
        self.authorize,self.clock,self.max_age = authorize,clock,max_age
        owner = digest("JOURNAL",registry.fingerprint,identity,auth_public(auth_key))
        self.journal = NonceJournal(journal_path,owner)
        self.sessions = {}
        self.mutex = threading.RLock()

    def _session(self,c,state):
        st = self.sessions.get(c["sid"])
        require(st is not None and st["context"] == c and st["state"] == state,"Unexpected local session state")
        return st

    def _abort(self,c):
        self.journal.abort(c)
        st = self.sessions.pop(c["sid"],None)
        if st:
            st.pop("k",None); st.pop("rho",None)

    def commit(self,context):
        """Steps 1-2: authorize, reserve the session, encrypt a fresh nonce point."""
        with self.mutex:
            c = clone(context)
            validate_context(self.registry,c)
            require(self.identity in c["signers"],"Signer not selected")
            require(abs(int(self.clock())-c["timestamp"]) <= self.max_age,"Stale timestamp")
            require(self.authorize(bytes.fromhex(c["message"]),clone(c)) is True,"Message not authorized")
            self.journal.reserve(c)
            try:
                k,rho = random_scalar(),random_scalar()
                ct = cs_encrypt(self.registry.record["cs"],label(c,self.identity),k,rho)
                self.sessions[c["sid"]] = dict(context=c,state="COMMITTED",k=k,rho=rho,ct=ct,Q=encode_point(k*G))
                return packet(self.key,"COMMIT",c,self.identity,ct)
            except Exception:
                self._abort(c); raise

    def lock(self,context,commits):
        """Step 3: authenticate and lock exactly one complete ciphertext list."""
        with self.mutex:
            c = clone(context)
            st = self._session(c,"COMMITTED")
            try:
                data = check_packets(self.registry,c,commits,"COMMIT")
                for i in c["signers"]:
                    cs_aux(self.registry.record["cs"],label(c,i),data[i])
                require(data[self.identity]==st["ct"],"Own commitment was replaced")
                ordered = [[i,data[i]] for i in c["signers"]]
                d = digest("LOCK",c,ordered)
                self.journal.transition(c,"COMMITTED","LOCKED",d=d)
                st.update(state="LOCKED",digest=d,ciphertexts=data)
                return packet(self.key,"LOCK",c,self.identity,dict(d=d))
            except Exception:
                self._abort(c); raise

    def reveal(self,context,locks):
        """Step 4: require all matching lock signatures before opening the nonce."""
        with self.mutex:
            c = clone(context)
            st = self._session(c,"LOCKED")
            try:
                data = check_packets(self.registry,c,locks,"LOCK")
                require(all(v==dict(d=st["digest"]) for v in data.values()),"Inconsistent lock certificate")
                pi = prove(self.registry.record["cs"],label(c,self.identity),st["digest"],self.identity,st["ct"],st["Q"],st["k"],st["rho"])
                self.journal.transition(c,"LOCKED","REVEALED")
                st["state"] = "REVEALED"
                return packet(self.key,"REVEAL",c,self.identity,dict(d=st["digest"],Q=st["Q"],proof=pi))
            except Exception:
                self._abort(c); raise

    def partial(self,context,reveals):
        """Step 5: verify proofs, consume the nonce, and issue a signed approval."""
        with self.mutex:
            c = clone(context)
            st = self._session(c,"REVEALED")
            try:
                data = check_packets(self.registry,c,reveals,"REVEAL")
                pk = self.registry.record["cs"]
                weights = lagrange(c["signers"])
                Q = O.copy()
                for i in c["signers"]:
                    v = data[i]; keys(v,("d","Q","proof"))
                    require(v["d"]==st["digest"],"Wrong reveal digest")
                    verify_proof(pk,label(c,i),st["digest"],i,st["ciphertexts"][i],v["Q"],v["proof"])
                    Q += weights[i]*decode_point(v["Q"],True)
                require(Q != O and int(Q.x)%N != 0,"Degenerate aggregate commitment")
                qe,r = encode_point(Q),int(Q.x)%N
                h = challenge(self.registry,c,st["digest"],qe)
                # Burn BEFORE computing or emitting the response. A crash here
                # leaves CONSUMING and requires a new session, never recomputation.
                self.journal.transition(c,"REVEALED","CONSUMING")
                k = st.pop("k"); st.pop("rho",None)
                response = (r*self.share+h*k)%N
                del k
                tau = auth_sign(self.key,"APPROVE",self.identity,approval(self.registry,c,st["digest"],qe))
                out = packet(self.key,"PARTIAL",c,self.identity,dict(d=st["digest"],Q=qe,s=scalar(response),approval=tau))
                self.journal.transition(c,"CONSUMING","CONSUMED",output=out)
                st.update(state="CONSUMED",reveals=data,aggregate=qe)
                return out
            except Exception:
                self._abort(c); raise

    def retransmit_partial(self,context):
        """Return the exact stored packet, including after a process restart."""
        return self.journal.cached(context)

    def abort(self,context):
        with self.mutex:
            self._abort(context)

    def aggregate(self,context,partials):
        """Step 6: reconstruct every partial point using one shared inverse."""
        with self.mutex:
            c = clone(context); st = self._session(c,"CONSUMED")
            data = check_packets(self.registry,c,partials,"PARTIAL")
            qe,d = st["aggregate"],st["digest"]
            Q = decode_point(qe,True); r = int(Q.x)%N
            h = challenge(self.registry,c,d,qe)
            # All partial and aggregate checks share the same public h and r.
            inverse_h = pow(h,-1,N)
            z2 = (-r*inverse_h)%N
            weights = lagrange(c["signers"])
            total,cert = 0,[]
            for i in c["signers"]:
                v = data[i]; keys(v,("d","Q","s","approval"))
                require(v["Q"]==qe and v["d"]==d,"Partial targets wrong payload")
                z = parse_scalar(v["s"])
                reconstructed = reconstruct_point(z,decode_point(self.registry.record["shares"][str(i)]),inverse_h,z2)
                require(reconstructed == decode_point(st["reveals"][i]["Q"],True),f"Invalid partial from signer {i}")
                auth_check(self.registry.record["auth"][str(i)],v["approval"],"APPROVE",i,approval(self.registry,c,d,qe))
                total = (total+weights[i]*z)%N
                cert.append(dict(identity=i,signature=v["approval"]))
            require(total != 0,"Zero aggregate response")
            signature = dict(context=c,d=d,Q=qe,s=scalar(total),approvals=cert)
            # The approvals and context were already validated above. Reuse the
            # inverse instead of repeating the full public verification procedure.
            R = reconstruct_point(total,decode_point(self.registry.record["P"],True),inverse_h,z2)
            require(R != O and int(R.x)%N == r and R == Q,"Invalid aggregate reconstruction")
            return signature


def reconstruct_point(response, public_point, inverse_h, z2):
    """Compute R = (s*h^{-1})G + (-r*h^{-1})P (paper Section 5).

    This low-level arithmetic helper is NOT a signature verifier. Callers must
    validate the point, scalars, context, and mandatory approval certificate.
    Partial responses may be zero; complete signature responses must not be.
    """
    require(type(response) is int and 0 <= response < N,"Invalid response")
    require(type(inverse_h) is int and 0 < inverse_h < N,"Invalid inverse")
    require(type(z2) is int and 0 <= z2 < N,"Invalid coefficient")
    return ((response*inverse_h)%N)*G + z2*public_point


def verify_or_raise(registry: Registry,message: bytes,signature,now=None,max_age=None):
    keys(signature,("context","d","Q","s","approvals"))
    c = signature["context"]
    validate_context(registry,c)
    require(type(message) is bytes and c["message"]==message.hex(),"Message mismatch")
    if max_age is not None:
        require(type(max_age) is int and max_age >= 0,"Invalid age policy")
        require(abs(int(time.time() if now is None else now)-c["timestamp"]) <= max_age,"Expired signature")
    strict_hex(signature["d"],32)
    Q = decode_point(signature["Q"],True)
    r,z = int(Q.x)%N,parse_scalar(signature["s"],True)
    require(r != 0,"Zero coordinate")
    h = challenge(registry,c,signature["d"],signature["Q"])
    # Phase 3: GOST-form reconstruction, with the full-point extension.
    inverse_h = pow(h,-1,N)
    z2 = (-r*inverse_h)%N
    R = reconstruct_point(z,decode_point(registry.record["P"],True),inverse_h,z2)
    require(R != O and int(R.x)%N == r and R == Q,"Invalid core reconstruction")
    approvals = signature["approvals"]
    require(type(approvals) is list and len(approvals)==len(c["signers"]),"Certificate required")
    M = approval(registry,c,signature["d"],signature["Q"])
    for i,a in zip(c["signers"],approvals):
        keys(a,("identity","signature"))
        require(type(a["identity"]) is int and a["identity"]==i,"Incorrect certificate ordering")
        auth_check(registry.record["auth"][str(i)],a["signature"],"APPROVE",i,M)


def verify(registry,message,signature,now=None,max_age=None) -> bool:
    try:
        verify_or_raise(registry,message,signature,now,max_age)
        return True
    except (ValueError,TypeError,KeyError,OverflowError,AttributeError):
        return False


def run_session(signers: list[Signer],context):
    """In-process message delivery harness, not a network/consensus service."""
    require([u.identity for u in signers]==context["signers"],"Wrong participants")
    try:
        commits = [u.commit(context) for u in signers]
        locks = [u.lock(context,commits) for u in signers]
        reveals = [u.reveal(context,locks) for u in signers]
        partials = [u.partial(context,reveals) for u in signers]
        results = [u.aggregate(context,partials) for u in signers]
        require(all(x==results[0] for x in results),"Inconsistent final signatures")
        return results[0],dict(commits=commits,locks=locks,reveals=reveals,partials=partials)
    except Exception:
        for u in signers:
            u.abort(context)
        raise
