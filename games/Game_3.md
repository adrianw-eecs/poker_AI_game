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
    [BTN     ] Player1 (YOU)  stack=1000  [4♣ 8♦]  (<<< action)
    [SB      ] Player2  stack=995  [     ]
    [BB      ] Player3  stack=990  [     ]

  Actions: f=fold  |  c=call(10)  |  r N=raise(20-1000)  |  a=all-in(1000)
  Your action: f
=== Hand #1  |  Street: PREFLOP ===

  Board: [ - ]
  Pot:   15  (0 + 10 + 5)
  To call: 10

  Players:
    [BTN     ] Player1  stack=1000  [     ] — folded  (FOLDED)
    [SB      ] Player2 (YOU)  stack=995  [9♣ 2♣]  (<<< action)
    [BB      ] Player3  stack=990  [     ]

  Action this street:
    Player1 (seat 0): FOLD

  Actions: f=fold  |  c=call(10)  |  r N=raise(20-1000)  |  a=all-in(1000)
  Your action: r 100
=== Hand #1  |  Street: PREFLOP ===

  Board: [ - ]
  Pot:   15  (0 + 10 + 5)
  To call: 100

  Players:
    [BTN     ] Player1  stack=1000  [     ] — folded  (FOLDED)
    [SB      ] Player2  stack=900  [     ]
    [BB      ] Player3 (YOU)  stack=990  [6♥ A♥]  (<<< action)

  Action this street:
    Player1 (seat 0): FOLD

  Actions: f=fold  |  c=call(100)  |  r N=raise(190-1000)  |  a=all-in(1000)
  Your action: c
=== Hand #1  |  Street: FLOP ===

  Board: [ J♣ 6♣ T♣ ]
  Pot:   200  (0 + 200)

  Players:
    [BTN     ] Player1  stack=1000  [     ]  (FOLDED)
    [SB      ] Player2 (YOU)  stack=900  [9♣ 2♣]  (<<< action)
    [BB      ] Player3  stack=900  [     ]

  Actions: f=fold  |  k=check  |  r N=raise(10-900)  |  a=all-in(900)
  Your action: k
=== Hand #1  |  Street: FLOP ===

  Board: [ J♣ 6♣ T♣ ]
  Pot:   200  (0 + 200)

  Players:
    [BTN     ] Player1  stack=1000  [     ]  (FOLDED)
    [SB      ] Player2  stack=900  [     ] — checked
    [BB      ] Player3 (YOU)  stack=900  [6♥ A♥]  (<<< action)

  Action this street:
    Player2 (seat 1): CHECK

  Actions: f=fold  |  k=check  |  r N=raise(10-900)  |  a=all-in(900)
  Your action: k
=== Hand #1  |  Street: TURN ===

  Board: [ J♣ 6♣ T♣ K♦ ]
  Pot:   200  (0 + 200)

  Players:
    [BTN     ] Player1  stack=1000  [     ]  (FOLDED)
    [SB      ] Player2 (YOU)  stack=900  [9♣ 2♣]  (<<< action)
    [BB      ] Player3  stack=900  [     ]

  Actions: f=fold  |  k=check  |  r N=raise(10-900)  |  a=all-in(900)
  Your action: k
=== Hand #1  |  Street: TURN ===

  Board: [ J♣ 6♣ T♣ K♦ ]
  Pot:   200  (0 + 200)

  Players:
    [BTN     ] Player1  stack=1000  [     ]  (FOLDED)
    [SB      ] Player2  stack=900  [     ] — checked
    [BB      ] Player3 (YOU)  stack=900  [6♥ A♥]  (<<< action)

  Action this street:
    Player2 (seat 1): CHECK

  Actions: f=fold  |  k=check  |  r N=raise(10-900)  |  a=all-in(900)
  Your action: k
=== Hand #1  |  Street: RIVER ===

  Board: [ J♣ 6♣ T♣ K♦ T♦ ]
  Pot:   200  (0 + 200)

  Players:
    [BTN     ] Player1  stack=1000  [     ]  (FOLDED)
    [SB      ] Player2 (YOU)  stack=900  [9♣ 2♣]  (<<< action)
    [BB      ] Player3  stack=900  [     ]

  Actions: f=fold  |  k=check  |  r N=raise(10-900)  |  a=all-in(900)
  Your action: k
=== Hand #1  |  Street: RIVER ===

  Board: [ J♣ 6♣ T♣ K♦ T♦ ]
  Pot:   200  (0 + 200)

  Players:
    [BTN     ] Player1  stack=1000  [     ]  (FOLDED)
    [SB      ] Player2  stack=900  [     ] — checked
    [BB      ] Player3 (YOU)  stack=900  [6♥ A♥]  (<<< action)

  Action this street:
    Player2 (seat 1): CHECK

  Actions: f=fold  |  k=check  |  r N=raise(10-900)  |  a=all-in(900)
  Your action: k

==================================================
SHOWDOWN
==================================================

Board: [ J♣ 6♣ T♣ K♦ T♦ ]

  Player1              (BTN     ): [4♣ 8♦]  -> folded
  Player2              (SB      ): [9♣ 2♣]  -> Flush, Jack-high  (WINNER: +100)
  Player3              (BB      ): [6♥ A♥]  -> Two pair, Tens and Sixs, Ace kicker

  Hand result: Player1 breaks even
  Hand result: Player2 wins 100 chips
  Hand result: Player3 loses 100 chips


  Hand result: Human  reward=+0.000


  Hand result: Human  reward=+0.200


  Hand result: Human  reward=+0.000

=== Hand #2  |  Street: PREFLOP ===