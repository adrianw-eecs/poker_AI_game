PS D:\Claude_Projects\poker> python -m poker.main -n 3 --bots human human human
============================================================
Texas Hold'em Poker Engine
============================================================
Players: 3
Starting stack: 1000
Blinds: 5/10
============================================================

=== Hand #1  |  Street: PREFLOP ===

  Board: [ - ]
  Pot:   15  (0 + 10 + 5)
  To call: 10

  Players:
    [BTN     ] Player1 (YOU)  stack=1000  [8♥ 6♥]  (<<< action)
    [SB      ] Player2  stack=995  [     ]
    [BB      ] Player3  stack=990  [     ]

  Actions: f=fold  |  c=call(10)  |  r N=raise(20-1000)  |  a=all-in(1000)
  Your action: r 500
=== Hand #1  |  Street: PREFLOP ===

  Board: [ - ]
  Pot:   15  (0 + 10 + 5)
  To call: 500

  Players:
    [BTN     ] Player1  stack=500  [     ]
    [SB      ] Player2 (YOU)  stack=995  [T♦ 4♠]  (<<< action)
    [BB      ] Player3  stack=990  [     ]

  Actions: f=fold  |  c=call(500)  |  r N=raise(990-1000)  |  a=all-in(1000)
  Your action: c
=== Hand #1  |  Street: PREFLOP ===

  Board: [ - ]
  Pot:   15  (0 + 10 + 5)
  To call: 500

  Players:
    [BTN     ] Player1  stack=500  [     ]
    [SB      ] Player2  stack=500  [     ] — called 500
    [BB      ] Player3 (YOU)  stack=990  [A♥ 3♠]  (<<< action)

  Action this street:
    Player2 (seat 1): CALL 500

  Actions: f=fold  |  c=call(500)  |  r N=raise(990-1000)  |  a=all-in(1000)
  Your action: c
=== Hand #1  |  Street: FLOP ===

  Board: [ 3♦ 2♥ 9♦ ]
  Pot:   1500

  Players:
    [BTN     ] Player1  stack=500  [     ]
    [SB      ] Player2 (YOU)  stack=500  [T♦ 4♠]  (<<< action)
    [BB      ] Player3  stack=500  [     ]

  Actions: f=fold  |  k=check  |  r N=raise(10-500)  |  a=all-in(500)
  Your action: k
=== Hand #1  |  Street: FLOP ===

  Board: [ 3♦ 2♥ 9♦ ]
  Pot:   1500

  Players:
    [BTN     ] Player1  stack=500  [     ]
    [SB      ] Player2  stack=500  [     ] — checked
    [BB      ] Player3 (YOU)  stack=500  [A♥ 3♠]  (<<< action)

  Action this street:
    Player2 (seat 1): CHECK

  Actions: f=fold  |  k=check  |  r N=raise(10-500)  |  a=all-in(500)
  Your action: k
=== Hand #1  |  Street: FLOP ===

  Board: [ 3♦ 2♥ 9♦ ]
  Pot:   1500

  Players:
    [BTN     ] Player1 (YOU)  stack=500  [8♥ 6♥]  (<<< action)
    [SB      ] Player2  stack=500  [     ] — checked
    [BB      ] Player3  stack=500  [     ] — checked

  Action this street:
    Player2 (seat 1): CHECK
    Player3 (seat 2): CHECK

  Actions: f=fold  |  k=check  |  r N=raise(10-500)  |  a=all-in(500)
  Your action: k
=== Hand #1  |  Street: TURN ===

  Board: [ 3♦ 2♥ 9♦ J♦ ]
  Pot:   1500

  Players:
    [BTN     ] Player1  stack=500  [     ]
    [SB      ] Player2 (YOU)  stack=500  [T♦ 4♠]  (<<< action)
    [BB      ] Player3  stack=500  [     ]

  Actions: f=fold  |  k=check  |  r N=raise(10-500)  |  a=all-in(500)
  Your action: k
=== Hand #1  |  Street: TURN ===

  Board: [ 3♦ 2♥ 9♦ J♦ ]
  Pot:   1500

  Players:
    [BTN     ] Player1  stack=500  [     ]
    [SB      ] Player2  stack=500  [     ] — checked
    [BB      ] Player3 (YOU)  stack=500  [A♥ 3♠]  (<<< action)

  Action this street:
    Player2 (seat 1): CHECK

  Actions: f=fold  |  k=check  |  r N=raise(10-500)  |  a=all-in(500)
  Your action: k
=== Hand #1  |  Street: TURN ===

  Board: [ 3♦ 2♥ 9♦ J♦ ]
  Pot:   1500

  Players:
    [BTN     ] Player1 (YOU)  stack=500  [8♥ 6♥]  (<<< action)
    [SB      ] Player2  stack=500  [     ] — checked
    [BB      ] Player3  stack=500  [     ] — checked

  Action this street:
    Player2 (seat 1): CHECK
    Player3 (seat 2): CHECK

  Actions: f=fold  |  k=check  |  r N=raise(10-500)  |  a=all-in(500)
  Your action: k
=== Hand #1  |  Street: RIVER ===

  Board: [ 3♦ 2♥ 9♦ J♦ J♠ ]
  Pot:   1500

  Players:
    [BTN     ] Player1  stack=500  [     ]
    [SB      ] Player2 (YOU)  stack=500  [T♦ 4♠]  (<<< action)
    [BB      ] Player3  stack=500  [     ]

  Actions: f=fold  |  k=check  |  r N=raise(10-500)  |  a=all-in(500)
  Your action: k
=== Hand #1  |  Street: RIVER ===

  Board: [ 3♦ 2♥ 9♦ J♦ J♠ ]
  Pot:   1500

  Players:
    [BTN     ] Player1  stack=500  [     ]
    [SB      ] Player2  stack=500  [     ] — checked
    [BB      ] Player3 (YOU)  stack=500  [A♥ 3♠]  (<<< action)

  Action this street:
    Player2 (seat 1): CHECK

  Actions: f=fold  |  k=check  |  r N=raise(10-500)  |  a=all-in(500)
  Your action: k
=== Hand #1  |  Street: RIVER ===

  Board: [ 3♦ 2♥ 9♦ J♦ J♠ ]
  Pot:   1500

  Players:
    [BTN     ] Player1 (YOU)  stack=500  [8♥ 6♥]  (<<< action)
    [SB      ] Player2  stack=500  [     ] — checked
    [BB      ] Player3  stack=500  [     ] — checked

  Action this street:
    Player2 (seat 1): CHECK
    Player3 (seat 2): CHECK

  Actions: f=fold  |  k=check  |  r N=raise(10-500)  |  a=all-in(500)
  Your action: k

==================================================
SHOWDOWN
==================================================

Board: [ 3♦ 2♥ 9♦ J♦ J♠ ]

  Player1              (BTN     ): [8♥ 6♥]  -> Pair of Jacks, Nine kicker
  Player2              (SB      ): [T♦ 4♠]  -> Pair of Jacks, Ten kicker
  Player3              (BB      ): [A♥ 3♠]  -> Two pair, Jacks and Threes, Ace kicker  (WINNER: +1000)

  Hand result: Player1 loses 500 chips
  Hand result: Player2 loses 500 chips
  Hand result: Player3 wins 1000 chips


  Hand result: Human  reward=+0.000


  Hand result: Human  reward=+0.000


  Hand result: Human  reward=+1.500

=== Hand #2  |  Street: PREFLOP ===