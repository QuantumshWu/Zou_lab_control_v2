"""Where a pulse server listens, and where a client looks for one.

Said once, and said HERE rather than inside the server, for two reasons.

The number itself was said five times in four packages -- the server default,
the client default, the CLI default, a widget's placeholder and an apparatus
form default -- so changing the port meant finding them all.

The server implementation is not the place to keep shared connection defaults:
importing the package merely to read a port must not load the socket server.
The product manifest imports that server only when ``zlc pulse_server`` is
selected.
"""

from __future__ import annotations


__all__ = [
    "DEFAULT_BIND_HOST",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_REQUEST_TIMEOUT",
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
