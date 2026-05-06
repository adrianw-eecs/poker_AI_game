PS D:\Claude_Projects\poker> python -m poker.main -n 4 --bots human flop_bot random flop_bot
============================================================
Texas Hold'em Poker Engine
============================================================
Players: 4
Starting stack: 1000
Blinds: 5/10
============================================================

=== Hand #1  |  Street: PREFLOP ===

  Board: [ - ]
  Pot:   15  (0 + 5 + 10 + 10)
  To call: 10

  Players:
    [BTN     ] Player1 (YOU)  stack=1000  [6♠ 8♣]  (<<< action)
    [SB      ] Player2  stack=995  [     ]
    [BB      ] Player3  stack=990  [     ]
    [UTG     ] Player4  stack=990  [     ] — called 10

  Action this street:
    Player4 (seat 3): CALL 10

  Actions: f=fold  |  c=call(10)  |  r N=raise(20-1000)  |  a=all-in(1000)
  Your action: f

==================================================
SHOWDOWN
==================================================

Board: [ A♦ J♣ J♦ Q♣ 6♥ ]

  Player1              (BTN     ): [6♠ 8♣]  -> folded
  Player2              (SB      ): [9♠ 8♠]  -> Pair of Jacks  (WINNER: +560)
  Player3              (BB      ): [6♣ Q♥]  -> folded
  Player4              (UTG     ): [2♥ 9♦]  -> Pair of Jacks  (WINNER: +410)

  Hand result: Player1 breaks even
  Hand result: Player2 wins 560 chips
  Hand result: Player3 loses 970 chips
  Hand result: Player4 wins 410 chips

=== Hand #2  |  Street: PREFLOP ===

  Board: [ - ]
  Pot:   15  (0 + 0 + 5 + 10)
  To call: 10

  Players:
    [UTG     ] Player1 (YOU)  stack=1000  [A♥ 7♦]  (<<< action)
    [BTN     ] Player2  stack=1000  [     ]
    [SB      ] Player3  stack=995  [     ]
    [BB      ] Player4  stack=990  [     ]

  Actions: f=fold  |  c=call(10)  |  r N=raise(20-1000)  |  a=all-in(1000)
  Your action: r 100
=== Hand #2  |  Street: PREFLOP ===

  Board: [ - ]
  Pot:   15  (100 + 100 + 25 + 100)
  To call: 100

  Players:
    [UTG     ] Player1 (YOU)  stack=900  [A♥ 7♦]  (<<< action)
    [BTN     ] Player2  stack=900  [     ] — called 100
    [SB      ] Player3  stack=975  [     ] — called 25
    [BB      ] Player4  stack=900  [     ] — called 100

  Action this street:
    Player2 (seat 1): CALL 100
    Player3 (seat 2): CALL 25
    Player4 (seat 3): CALL 100

  Actions: f=fold  |  c=call(100)  |  r N=raise(190-1000)  |  a=all-in(1000)
  Your action: c
=== Hand #2  |  Street: PREFLOP ===

  Board: [ - ]
  Pot:   15  (100 + 100 + 5 + 100)
  To call: 100

  Players:
    [UTG     ] Player1 (YOU)  stack=900  [A♥ 7♦] — called 100  (<<< action)
    [BTN     ] Player2  stack=900  [     ] — called 100
    [SB      ] Player3  stack=995  [     ] — called 5
    [BB      ] Player4  stack=900  [     ] — called 100

  Action this street:
    Player2 (seat 1): CALL 100
    Player3 (seat 2): CALL 25
    Player4 (seat 3): CALL 100
    Player1 (seat 0): CALL 100
    Player2 (seat 1): CALL 100
    Player3 (seat 2): CALL 5
    Player4 (seat 3): CALL 100

  Actions: f=fold  |  c=call(100)  |  r N=raise(190-1000)  |  a=all-in(1000)
  Your action: c
Error: Raise/all-in amount 30 must exceed bet to call 100
PS D:\Claude_Projects\poker> 