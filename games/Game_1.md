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
    [BTN     ] Player1 (YOU)  stack=1000  [2♥ 2♦]  (<<< action)
    [SB      ] Player2  stack=995  [     ]
    [BB      ] Player3  stack=990  [     ]

  Actions: f=fold  |  c=call(10)  |  r N=raise(20-1000)  |  a=all-in(1000)
  Your action: a
=== Hand #1  |  Street: PREFLOP ===

  Board: [ - ]
  Pot:   15  (0 + 10 + 5)
  To call: 1000

  Players:
    [BTN     ] Player1  stack=0  [     ] — all-in 1000  (ALL-IN)
    [SB      ] Player2 (YOU)  stack=995  [9♠ Q♣]  (<<< action)
    [BB      ] Player3  stack=990  [     ]

  Action this street:
    Player1 (seat 0): ALL-IN 1000

  Actions: f=fold  |  c=call(995)  |  a=all-in(1000)
  Your action: c
Error: Call amount 995 is less than bet to call 1000
PS D:\Claude_Projects\poker>