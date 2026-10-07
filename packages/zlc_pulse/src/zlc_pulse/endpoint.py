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
machine's clients from a peer's, listing the addresses a peer can reach,
dropping a connection, and the rule every listening socket binds by (the
device fabric's announcer keeps it too) -- live here as well, for the same
reason: the SLM server needs them without loading the pulse server.  The
one spelling of the machine a client dials lives beside the test that tells
this machine from a peer, which it is built on: a pulse board and a knob
the device fabric serves are both named by it, so one machine written two
ways is one device to the broker.  The Device Manager's Remote asks the
same rule, unresolved, whether an endpoint is this machine's to publish.
"""

from __future__ import annotations

from collections.abc import Iterable
import errno
import ipaddress
import os
import socket


__all__ = [
    "DEFAULT_BIND_HOST",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_REQUEST_TIMEOUT",
    "bind_exclusive",
    "dialled_address",
    "drop_connection",
    "drop_peer_connections",
    "is_loopback_host",
    "is_this_machine",
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


def bind_exclusive(sock: socket.socket, address: tuple[str, int]) -> None:
    """Bind a server's socket as the only one on its port, or say why not.

    The rule every listening socket here binds by.  A second listener on a
    held port must be refused at its bind -- on the wildcard or on the same
    address.  Two different specific addresses, 127.0.0.1 and 127.0.0.2,
    may still share a port: each client reaches the one it dialled, and
    every bench server binds the wildcard.
    Windows refuses a plain bind only on the very same address: a listener
    on 127.0.0.1 binds beside one on 0.0.0.0 and, as the more specific one,
    takes every later loopback connection, so a second bench's server takes
    the first bench's own clients and its board.  SO_EXCLUSIVEADDRUSE
    refuses that bind in either order and still rebinds a port whose server
    has closed; SO_REUSEADDR would instead let a second socket bind a port a
    live listener holds.  POSIX keeps SO_REUSEADDR for what it means there,
    rebinding past TIME_WAIT -- on a stream socket only: on a datagram one it
    lets two bind the same port.  Linux still refuses a specific address
    beside a wildcard listener with it; BSD and macOS let that bind through,
    so the overlap above is refused on Windows and Linux only.  The refusal
    names the port, not the socket layer's words for it (Windows answers
    "forbidden by its access permissions" when a wildcard listener holds
    the port).
    """

    if os.name == "nt":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    elif sock.type == socket.SOCK_STREAM:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(address)
    except OSError as error:
        if error.errno not in (errno.EADDRINUSE, errno.EACCES):
            raise
        raise OSError(
            error.errno,
            f"port {address[1]} is taken: another server on this machine already "
            f"listens on it, or the system reserves it ({error.strerror})",
        ) from error


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


def is_this_machine(host: str, local_addresses: Iterable[str] = ()) -> bool:
    """Whether ``host``, as written, is this machine -- never resolved.

    An empty host (a dial to it reaches this machine or nothing),
    ``localhost``, this machine's own name (``socket.gethostname``, which
    asks no resolver), a loopback address (all of 127/8, ``::1`` and their
    mapped forms), or one of ``local_addresses``: this machine's own LAN
    addresses as ``local_ipv4_addresses`` lists them, which the caller asks
    for where a resolver that does not answer freezes nothing -- listing
    them resolves this machine's name.  Without them only the loopback
    spellings and the name are this machine.  Any other name is not
    resolved here: ``dialled_address`` resolves it, then asks this of the
    address.
    """

    name = str(host).strip().lower()
    return (
        not name
        or name == "localhost"
        or name == socket.gethostname().strip().lower()
        or is_loopback_host(name)
        or name in local_addresses
    )


def dialled_address(host: str) -> str:
    """The machine a client dialling ``host`` reaches, in one spelling.

    Every spelling of this machine (``is_this_machine``) is ``127.0.0.1``,
    and any other name is resolved to its first IPv4 address, so one
    server written down as its name and as its address is one server.  A
    name that does not resolve -- or that cannot even be asked, an empty or
    an over-long label -- reaches nothing and is kept as written: the dial
    is what reports it.  A name costs the resolver, so this belongs where a
    connection is set up, not on a request.
    """

    name = str(host).strip().lower()
    if is_this_machine(name):
        return "127.0.0.1"
    try:
        address = str(ipaddress.ip_address(name))
    except ValueError:
        try:
            address = socket.getaddrinfo(
                name, None, socket.AF_INET, socket.SOCK_STREAM
            )[0][4][0]
        except (OSError, IndexError, UnicodeError):
            return name
    if is_this_machine(address, local_ipv4_addresses()):
        return "127.0.0.1"
    return address


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
