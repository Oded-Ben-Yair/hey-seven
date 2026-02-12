# Casino Comp System: Rules and Formulas

## Core Principle

Comps are based on THEORETICAL loss (expected loss), NOT actual win/loss. This means a player who wins $10,000 in a session still earns comps based on what the casino expected to win from their action.

## ADT Formula

```
ADT = Average Bet × Decisions Per Hour × Hours Played × House Edge
```

ADT stands for Average Daily Theoretical. It represents the predicted average loss a player sustains per day of play.

## Game Parameters

### Decisions Per Hour
- Blackjack: 60-80 hands per hour
- Baccarat: 70-90 hands per hour
- Craps: 100-120 rolls per hour
- Roulette: 35-45 spins per hour
- Slots: 500-700 spins per hour
- Video Poker: 500-600 hands per hour

### House Edge Ranges
- Blackjack: 0.5% (basic strategy) to 2.0% (average player)
- Baccarat banker: 1.06%
- Baccarat player: 1.24%
- Craps pass line: 1.41%
- Roulette (American double-zero): 5.26%
- Roulette (European single-zero): 2.7%
- Slots: 2-15% (varies by machine and casino)
- Video Poker: 0.5-5% (varies by paytable)

## Calculation Examples

### Example 1: Blackjack Player
Average bet: $50
Decisions per hour: 70
Hours played: 4
House edge: 1.5%
ADT = $50 × 70 × 4 × 0.015 = $210

### Example 2: Slot Player
Average bet per spin: $2
Spins per hour: 600
Hours played: 5
House edge: 8%
ADT = $2 × 600 × 5 × 0.08 = $480

### Example 3: Baccarat High-Roller
Average bet: $100
Decisions per hour: 80
Hours played: 5
House edge: 1.06%
ADT = $100 × 80 × 5 × 0.0106 = $424

### Example 4: Roulette Player
Average bet: $20
Spins per hour: 30
Hours played: 4
House edge: 5.26%
Total wager = $20 × 30 × 4 = $2,400
ADT = $2,400 × 0.0526 = $126.24

## Slot Points Conversion

Common structure: $4 wagered = 1 club point. 100 points = $1 in cashback or free play. This equals 0.25% reward rate ($400 play = $1 reward).

Video poker typically offers lower reward rates (0.15% or less) due to higher payback percentages.

## Reinvestment Rates

The reinvestment rate is the percentage of theoretical win returned to the player as comps.

| Player Tier | Reinvestment Rate | ADT Range |
|------------|------------------|-----------|
| Casual/Entry | 10-15% | Under $100 |
| Mid-tier | 15-20% | $100-$500 |
| Premium | 20-30% | $500-$2,000 |
| High-roller | 30-40% | $2,000+ |

Typical marketing department average: ~20% of theoretical win.

Rates fluctuate with economic conditions and competitive pressure. Strong economy: 25-35%. Downturns: 5-15%.

## Comp Calculation Example

Player ADT: $500
Reinvestment rate: 25%
Daily comp budget: $500 × 0.25 = $125
3-day trip comp budget: $125 × 3 = $375

This $375 might be allocated as: $150 room credit + $100 dining + $75 free play + $50 show tickets.

## Comp Types (Hierarchy, Lowest to Highest Value)

1. Complimentary beverages: $2-5 per drink, virtually all active players
2. Meal comps: Casual $10-20, mid-tier $25-50, premium $100-500
3. Hotel room comps: 10-20% discount, free standard, free suite ($1,000-5,000/night)
4. Show and entertainment tickets: Discounted to VIP seating
5. Ground transportation: Valet, limo service ($50-200)
6. Airfare: Economy to first class
7. Private jet service: Ultra-premium only (tens of thousands)
8. Loss rebates: 5-10% of net loss returned as cash
9. Cashback: Direct currency rebate on theoretical

## Comp Approval Authority

| Level | Approval Range | Who |
|-------|---------------|-----|
| Frontline | Under $100 | Pit supervisor |
| Mid-level | $100-$500 | Pit boss/manager |
| Host | $500-$1,000 | Casino host |
| Senior | $1,000+ | Marketing/casino management |

Comp exceptions (outside matrix rules) require documented business rationale and manager sign-off. Most casinos require two-person sign-off above specified thresholds.

## Real-Time vs. Pre-Arranged Comps

### Real-time comps
Offered during or immediately after a visit based on that session's action. Slot comps: automatic via card tracking. Table comps: player requests from pit supervisor based on observed play. Provides immediate recognition and positive reinforcement.

### Pre-arranged marketing comps (bounce-back offers)
Delivered 1-3 months after visit via mail, email, or app. Based on historical data analysis. Targeted offers calibrated to reinvestment rates. Often time-limited to drive near-term return visits.

### Hybrid approach
Real-time: Earn $30 free play during visit (immediate gratification).
Post-visit: Receive $50 free play bounce-back offer via email one week later (drives return).

## Trip-Level vs. Cumulative Theoretical

### Trip-level theoretical
Used for: individual visit comp packages, bounce-back offers, near-term competitive positioning.
Calculation period: single visit.

### Cumulative theoretical (rolling 12 months)
Used for: tier status determination, annual benefits, long-term comp eligibility.
Calculation period: typically rolling 12 months.

### Best practice
Use trip-level for promotional comps (drive near-term visits). Use cumulative for tier status and long-term benefits. This combination optimizes both short-term competitiveness and long-term profitability.

## ADT Threshold for Host Assignment

Players with ADT of $300-500+ typically qualify for a dedicated casino host. Below this threshold, players receive automated marketing and loyalty program benefits.

## Important Metric Variations

### ADW (Average Daily Worth)
Better of ADT or a percentage (often 40%) of actual loss. Addresses cases where players lose significantly in short sessions (low theoretical but high actual).

### ATT (Average Trip Theoretical)
Total theoretical per trip rather than per day. Used primarily for cruise casino offers.

### Total Guest Value (TGV)
Gaming theoretical + non-gaming spend (hotel, F&B, retail, entertainment). Provides holistic player value assessment.