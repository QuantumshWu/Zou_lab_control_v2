"""Where a pulse server listens, and where a client looks for one.

Said once, and said HERE rather than inside the server, for two reasons.

The number itself was said five times in four packages -- the server default,
the client default, the CLI default, a widget's placeholder and an apparatus
form default -- so changing the port meant finding them all.

The server implementation is not the place to keep shared connection defaults:
importing the package merely to read a port must not load the socket server.
The product manifest imports that server only when ``zlc pulse_server`` is
selected.

The socket chores the pulse server and the SLM server share -- telling this
machine's clients from a peer's, listing the addresses a peer can reach, and
dropping a connection -- live here too, for the same reason: the SLM server
needs them without loading the pulse server.
"""

from __future__ import annotations

from collections.abc import Iterable
import ipaddress
import socket


__all__ = [
    "DEFAULT_BIND_HOST",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_REQUEST_TIMEOUT",
    "drop_connection",
    "drop_peer_connections",
    "is_loopback_host",
    "local_ipv4_addresses",
]

#: The port a pulse server listens on and a client dials.
DEFAULT_PORT = 18861

#: What a server binds by default: every interface, because the board is often
#: on a different machine from the operator.
DEFAULT_BIND_HOST = "0.0.0.0"

#: Where a client looks by default: this machine, because it usually is.
DEFAULT_HOST = "127.0.0.1"

#: How long a client waits for the server to ANSWER.  A request can be
#: slow for honest reasons -- the board is mid-shot and holding its lock,
#: a session is warming up -- so this is generous.  It used to be written
#: down three other places instead: the server prints 30 in the connect
#: example it hands operators, the apparatus device passes 30, and the
#: client class defaulted to 5 -- which is the one the pulse editor got.
DEFAULT_REQUEST_TIMEOUT = 30.0

#: How long a client waits to REACH the server at all.  A different
#: question with a different answer: a listening port answers a connection
#: on a LAN in milliseconds, so waiting longer only delays telling the
#: operator that nothing is there -- which is what a dropped SYN looks
#: like, and a firewall drops rather than refuses.
DEFAULT_CONNECT_TIMEOUT = 5.0


def is_loopback_host(host: str) -> bool:
    """Whether ``host`` is THIS machine talking to itself.

    A server that listens on every interface sees its own machine's
    clients arrive from 127.0.0.0/8 or ``::1`` (or the IPv4-mapped form of
    either), and a peer's from that peer's address; this is the one test
    both the pulse and the SLM server apply to tell them apart.  Text that
    is not an address is not loopback.
    """

    try:
        address = ipaddress.ip_address(str(host).strip())
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return bool((mapped or address).is_loopback)


def local_ipv4_addresses() -> tuple[str, ...]:
    """Discover non-loopback IPv4 addresses without requiring a network request."""

    addresses: list[str] = []
    packed_addresses: set[bytes] = set()

    def add(value: object) -> None:
        try:
            address = str(value).strip()
            packed = socket.inet_aton(address)
        except (OSError, ValueError):
            return
        if address == "0.0.0.0" or address.startswith("127.") or packed in packed_addresses:
            return
        packed_addresses.add(packed)
        addresses.append(address)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            add(sock.getsockname()[0])
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
            add(info[4][0])
    except OSError:
        pass

    try:
        for address in socket.gethostbyname_ex(socket.gethostname())[2]:
            add(address)
    except OSError:
        pass

    return tuple(addresses)


def drop_connection(connection: socket.socket) -> None:
    """End a connection nobody wants any more, from this side.

    Windows can leave a blocking recv waiting after shutdown alone. Close
    the revoked socket here instead of waiting for its peer to speak again.
    """

    try:
        connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    finally:
        connection.close()


def drop_peer_connections(connections: Iterable[socket.socket]) -> int:
    """Drop every connection from another machine; return how many went.

    What a server does when it stops being on offer: this machine's own
    clients stay connected.
    """

    dropped = 0
    for connection in connections:
        try:
            peer = connection.getpeername()[0]
        except OSError:
            continue
        if not is_loopback_host(peer):
            drop_connection(connection)
            dropped += 1
    return dropped
