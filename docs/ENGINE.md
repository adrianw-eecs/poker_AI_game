# Poker Engine Design & Architecture

**Document Purpose**: Technical deep-dive into the poker engine design, data structures, and algorithms

---

## Core Philosophy

**Immutability + Determinism**: All game state is immutable (frozen dataclasses). Each action produces a new state via `replace()`, enabling:
- Deterministic replay from event log
- Easy testing and validation
- Clear state transitions
- No hidden mutations

**Separation of Concerns**:
- **Engine** (hand/session): Orchestration and flow control
- **State** (GameState, PlayerState, Pot): Data structure and queries
- **Validation** (ActionValidator): Determine legal actions
- **Evaluation** (Evaluator): Hand strength ranking
- **Bots** (Protocol): Pluggable decision makers

---

## Data Structures

### GameState (Immutable Core)
```python
@dataclass(frozen=True)
class GameState:
    hand_number: int                           # 0-indexed hand count
    street: Street                             # PREFLOP/FLOP/TURN/RIVER/SHOWDOWN
    dealer_seat: int                           # Button position
    players: tuple[PlayerState, ...]           # Circular, all players
    
    # Betting state
    current_bet_to_call: int                   # Amount to call to stay in hand
    last_raise_size: int                       # Size of last raise (for min-raise)
    action_on_seat: int | None                 # Who acts next (None = round closed)
    
    # Card state
    hole_cards: tuple[Card, ...] | None        # Community cards (flop/turn/river)
    community_cards: tuple[Card, ...]
    
    # History
    action_history_this_street: list[(int, Action)]   # Actions on current street
    action_history_this_hand: list[(int, Action)]     # All actions in hand
    
    # Config & pots
    config: GameConfig
    pots: list[Pot]                            # Side pots (if applicable)
```

**Why Frozen?** Prevents accidental mutation. Every change creates new state via:
```python
state = replace(state, hand_number=1, street=Street.FLOP)  # Creates copy
```

### PlayerState (Individual Player)
```python
@dataclass(frozen=True)
class PlayerState:
    seat: int                   # 0-based position
    stack: int                  # Remaining chips
    hole_cards: tuple[Card, Card]
    
    # Commitment tracking
    committed_this_street: int  # Chips in pot this street (resets each street)
    committed_this_hand: int    # Total chips in pot this hand (never resets)
    
    # Status flags
    has_folded: bool
    is_all_in: bool             # Stack = 0 after this action
    is_eliminated: bool         # Permanent (stack was <= 0 at hand end)
```

**Commitment Distinction**:
- `committed_this_street`: Used for betting round logic (reset on new street)
- `committed_this_hand`: Used for pot calculation and showdown (never reset)

This enables:
- Quick "how much does player need to call?" → `current_bet_to_call - committed_this_street`
- Accurate pot distribution → sum of `committed_this_hand` across players

---

## Betting Round Logic

### BettingRound.run()

**Flow**:
1. Skip if ≤1 active players (round auto-closed)
2. Loop while `_round_is_closed()` returns False:
   - Get action from player
   - Apply action to state
   - Update `max_commitment` (highest bet seen)
   - Check if action re-opens betting (half-raise rule)
   - Log action event
3. Return state with `action_on_seat=None`

**Key Invariant: Round closes when all active players have:**
1. Had an opportunity to act at least once this street
2. Either matched `max_commitment` OR committed all available chips

### Half-Raise Rule (Re-open Betting)

**When does a bet/raise re-open the action?**

- `RAISE` action: **Always** re-opens
- `ALL_IN` action: Re-opens only if `raise_amount >= min_raise_increment`
  ```
  raise_amount = all_in_amount - current_bet_to_call
  min_raise_increment = max(big_blind, last_raise_size)
  re_opens = (raise_amount >= min_raise_increment)
  ```

**Example**:
- Current bet to call: $100
- Player A goes all-in for $130 (raise amount = $30)
- Min raise increment: $50
- **Result**: Doesn't re-open (short all-in)

Player B can just call and close the action.

### Side Pot Detection

**Problem**: Player has less chips than the bet they want to match
```
Seat 0: stack=100, committed=0, faces bet of 200
Seat 1: stack=500, committed=0

Seat 0 goes all-in for 100 (total committed=100)
Seat 1 commits 200
```

**Solution** in `_round_is_closed()`:
```python
for player in can_act_seats:
    # Check if they've matched max_commitment
    if player.committed_this_street >= max_commitment:
        continue  # Matched
    
    # Check if they COULD match if they had chips
    total_available = player.stack + player.committed_this_street
    if total_available >= max_commitment:
        return False  # Round stays open
    
    # They've committed ALL chips but less than max
    # This is OK for side pot; continue to next player
```

---

## Hand Execution (play_hand)

**Flow**:
```
1. Emit HandStarted event
2. Post antes (if any)
3. Post blinds (small blind, big blind)
4. Deal hole cards (2 per player)
5. Set action to UTG (or button in heads-up)
6. While ≤1 players remain in hand:
   a. Run betting round
   b. If ≤1 players remain, break (everyone folded)
   c. Deal next street (FLOP: 3 cards, TURN: 1, RIVER: 1)
   d. Reset street action (committed_this_street → 0)
7. Showdown: determine winner + distribute pot
8. Update stacks (including rake)
9. Mark players as eliminated if stack ≤ 0
10. Emit HandEnded event
```

### Showdown Logic (resolve)

**Single winner** (everyone else folded):
- No hand evaluation needed
- All chips go to folder winner (uncalled portion, no rake)

**Multi-way**:
1. Evaluate all remaining hands
2. Check if run-it-twice triggers:
   - Exactly 2 players remain
   - Both all-in
   - Streets remain to deal
3. Build pots (side pots if needed)
4. Apply rake
5. Distribute chips based on hand ranks

**Pot Distribution Algorithm**:
```python
# Build pots (only include chips from non-folded players)
# Award each pot to best hand that contributed to it
pots = build_pots(committed_by_seat, folded_seats)
pots_with_rake = apply_rake(pots, rake_percent, rake_cap)
awards = distribute(pots_with_rake, hand_ranks, dealer_seat)
```

---

## Session Management

### Session.run() Loop

**Flow**:
```
1. Initialize session (start time, rebuy tracking)
2. While not session_over:
   a. Check termination conditions:
      - Only 1 active player remains
      - Hand limit reached (duration_hands)
      - Time limit reached (duration_seconds)
   b. If terminating, break
   c. Play hand via play_hand()
   d. Apply rebuys (if enabled)
   e. Advance to next hand
3. Return final state
```

### Rebuy System

**When rebuy applies**:
```python
if session_config.rebuy_enabled:
    for player in state.players:
        if player.is_eliminated and player.stack == 0:
            # Reset to starting_stack
            player = player.with_stack(starting_stack)
            player = player.with_eliminated(False)
```

**Key invariant**: Rebuys happen **between hands**, not during. Players can't rebuy mid-hand.

---

## Action Validation

### Legal Actions (action_validator.py)

Determines what actions are available to a player:

```python
legal_actions(state, seat) → list[Action]
```

**Rule priorities**:
1. **If no chips**: Only fold available
2. **Check option**: If no bet to call, can check
3. **Call option**: If bet to call and chips available, can call
4. **Raise option**: If chips available, can raise (min-raise enforced)
5. **All-in**: If chips < amount needed for legal action, can go all-in

**Min-raise calculation**:
```
min_raise_amount = max(big_blind, last_raise_size)
min_total_to_raise_to = current_bet_to_call + min_raise_amount

# If player can't achieve min-raise, only valid action is all-in
if stack + committed_this_street < min_total_to_raise_to:
    legal_actions = [fold, call (if legal), all_in]
else:
    legal_actions = [fold, check (if legal), call, raise, all-in]
```

---

## Event Logging

### Event Types (events.py)

- **HandStarted**: Hand begins
- **BlindPosted**: Small/big blind posted
- **HoleCardsDealt**: Cards dealt to players
- **ActionTaken**: Player action (fold/check/call/raise/all-in)
- **BoardCardsDealt**: Community cards revealed
- **StreetEnded**: Betting round concluded
- **HandEnded**: Hand complete, stacks updated
- **TimingEvent**: Performance measurement

### Logging Implementation (logger.py)

**Buffering for performance**:
```python
class GameLogger:
    def log_event(self, event):
        # Just add to memory buffer (no I/O)
        self._buffer.append(json.dumps(event))
    
    def flush(self):
        # Batch write to disk
        with open(filepath, "a") as f:
            for line in self._buffer:
                f.write(line + "\n")
        self._buffer.clear()
```

**Result**: 30+ events per hand batched into 1-2 disk writes

---

## Bot Interface

### Bot Protocol
```python
class Bot(Protocol):
    def act(self, observation: GameView, legal_actions: list[Action]) -> Action:
        """Return action given view and legal actions."""
        ...
    
    def observe_result(self, observation: GameView, reward: float) -> None:
        """Notify bot of hand result (reward = stack delta / starting_stack)."""
        ...
```

### GameView (What Bots See)
```python
@dataclass(frozen=True)
class GameView:
    my_seat: int
    my_stack: int
    my_hole_cards: tuple[Card, Card]
    
    # Public info
    dealer_seat: int
    street: Street
    action_on_seat: int
    current_bet_to_call: int
    
    # Other players (no hole cards)
    other_players: tuple[PublicPlayerView, ...]
    
    # History
    action_history_this_street: list[(int, Action)]
    board_cards: tuple[Card, ...]
```

Bots can't see opponent hole cards until showdown.

---

## Performance Optimizations

### 1. Disable Diagnostics (5-10% speedup)
```python
# In SessionConfig
enable_diagnostics: bool = False  # Default: no console output
```

### 2. Lazy Showdown Evaluation (15-20% speedup)
```python
# Pre-convert community cards once
community_cards_list = list(state.community_cards)

# Reuse same list for all hand evaluations
for seat in non_folded:
    all_cards = list(player.hole_cards)
    all_cards.extend(community_cards_list)  # Extend, don't concatenate
    hand_ranks[seat] = evaluate(all_cards)
```

### 3. Event I/O Batching (20-30% speedup)
- Already implemented in GameLogger
- Buffer events in memory
- Flush to disk once per hand

### 4. Batch State Mutations (5-10% speedup)
- Collect all state updates
- Apply in single `replace()` call
- Reduces object allocations

---

## Testing Strategy

### Unit Tests
- Action validation
- Hand evaluation
- Pot distribution
- Elimination detection

### Integration Tests (Smoke)
- Multi-hand sessions
- Rebuy mechanics
- Player elimination
- Chip conservation

### End-to-End Tests
- 8-player 50-hand games
- Edge cases (side pots, run-it-twice)
- Performance benchmarks

---

## Common Issues & Solutions

### Issue: Infinite Loop in Betting Round
**Symptom**: Game hangs during hand 2-3  
**Cause**: `_round_is_closed()` doesn't recognize side pot  
**Solution**: Check `total_available = stack + committed_this_street`

### Issue: Hand Numbering Wrong
**Symptom**: Hand numbers cycle 0,0,1,0,0,1...  
**Cause**: Betting loop infinite loop prevents hand advancement  
**Solution**: Fix betting round logic

### Issue: Chips Lost
**Symptom**: Total chips end < total chips start  
**Cause**: Side pots not distributed properly  
**Solution**: Validate pot distribution in showdown

### Issue: Rebuy Not Working
**Symptom**: Game ends with 1 active player despite rebuy enabled  
**Cause**: Rebuy not being applied or eliminated flag not cleared  
**Solution**: Verify `_apply_rebuys()` is called and updates state

---

## Future Enhancements

1. **Multi-table support**: Split players across virtual tables
2. **Tournament structures**: Blind escalation, elimination tracking
3. **Hand history export**: Detailed hand-by-hand replay
4. **Equity calculation**: Real-time equity display for human play
5. **Position-based sizing**: Blind schedules that escalate
6. **Advanced betting rules**: Cap betting rounds, fold count rules
7. **Statistics**: Track win rates, hand histories, position stats

---

## References

- **GameState**: `src/poker/state/game_state.py`
- **Session**: `src/poker/engine/session.py`
- **Betting**: `src/poker/engine/betting_round.py`
- **Hand Flow**: `src/poker/engine/hand_engine.py`
- **Showdown**: `src/poker/engine/showdown.py`
- **Evaluation**: `src/poker/evaluation/evaluator.py`
