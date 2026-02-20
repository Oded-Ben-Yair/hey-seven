# Casino Domain Overview — Hey Seven Knowledge Foundation

**Last Updated**: 2026-02-12
**Sources**: Perplexity Research (6 streams), industry publications, UNLV gaming research, AGA reports

---

## 1. Casino VIP Host: Role and Daily Workflow

### Core Responsibilities
A casino VIP host manages relationships with high-value guests through personalized service, exclusive offers, and prompt issue resolution. The role splits into three buckets:
- **Relationship** (40-50%): Knowing player preferences, family details, business context
- **Operations** (30-35%): Comps, payouts, tournament invites, tier management
- **Compliance** (15-25%): KYC reviews, AML flags, documenting exceptions

### Time Allocation
| Activity | % of Shift | Notes |
|----------|-----------|-------|
| Floor operations (greeting, monitoring, service) | 70-85% | Standing/walking, interacting with guests |
| Digital tasks (CRM, email, SMS, reports) | 15-30% | Office/mobile device work |
| Morning data review | 5-10% | 40-60 min pre-shift prep |

### Player Portfolio
- **Typical portfolio size**: 300-450 players in database
- **Active attention**: 70-120 players receiving regular contact
- **Secondary tier**: 150-200 "active-fading" or "in development"
- **Tertiary group**: 50-150 "pending" or "reactivation" prospects
- **Top 5% of portfolio** contributes 30-40% of host's personal revenue

### Host Daily Routine

**Morning (Pre-shift, 40-60 min)**:
1. Review overnight reports and business intelligence dashboards
2. Scan player alerts: birthdays, declining activity, upcoming visits
3. Review Daily Action Plan (system-generated recommendations)
4. Prioritize: escalated issues > on-property guests > scheduled callbacks > proactive outreach

**Shift Operations**:
1. Floor time: greet VIPs, monitor gaming, real-time comp decisions
2. Digital blocks: email batches, SMS campaigns, CRM documentation
3. Call campaigns: contact 5-10 inactive players per shift
4. Comp processing: document decisions, get approvals for exceptions

**Evening/End of Shift**:
1. Document all player interactions in CRM
2. Prepare handoff notes for next shift
3. Confirm next-day guest arrivals and special requests

### Contact Frequency Targets
- Active players: 95% contacted each quarter
- Active-due-back and active-fading: monthly contact
- Pending and pending-inactive: quarterly contact
- Any guest absent 30+ days: proactive outreach to understand why

### Technology Stack
| System Type | Examples | Purpose |
|------------|---------|---------|
| Primary CRM | Salesforce (Caesars), PowerHost (Harvest Trends), QCI | Player management, daily action plans |
| Player Tracking | SYNKROS (Konami), ADVANTAGE (IGT), CasinoTrac | Real-time gaming data, ratings |
| Property Management | Agilysys, Infor, Opera | Hotel, F&B reservations |
| Mobile Host Apps | Gaming Analytics Mobile, Pillars PRM | Floor-based decisions, approvals |
| Workforce Scheduling | Casino Schedule Ease | Shift management |
| Compliance | Entegrity AML (IGT), custom systems | BSA/AML monitoring |

### Common Pain Points
1. **"Call-a-host-itis"**: Other departments route all guest issues to hosts
2. **Low-value comp requests**: Too much time on trivial comp decisions
3. **Communication vacuum**: Hosts not informed about offers/invites sent to their players
4. **Bonus program instability**: Constantly changing incentive structures
5. **Floor vs. office tension**: Management wants floor presence; development requires office time
6. **Disjointed systems**: CRM, payments, and compliance tools don't communicate
7. **Poor escalation SLAs**: Player complaints sitting in queues

---

## 2. Casino Comp System

### The Theoretical Foundation
Casinos award comps based on **theoretical loss** (expected loss), NOT actual win/loss. This ensures comps reflect player value regardless of short-term luck.

### ADT Formula (Average Daily Theoretical)
```
ADT = Average Bet × Decisions Per Hour × Hours Played × House Edge
```

**Decisions Per Hour by Game**:
| Game | Decisions/Hour | Typical House Edge |
|------|---------------|-------------------|
| Blackjack | 60-80 hands | 0.5-2.0% |
| Baccarat | 70-90 hands | 1.06-1.24% |
| Craps | 100-120 rolls | 1.4-16.7% (by bet type) |
| Roulette | 35-45 spins | 2.7-5.26% |
| Slots | 500-700 spins | 2-15% |
| Video Poker | 500-600 hands | 0.5-5% |

**Example Calculations**:
- **Blackjack**: $50 avg bet × 70 hands/hr × 4 hours × 1.5% edge = **$210 ADT**
- **Slots**: $2/spin × 600 spins/hr × 5 hours × 8% edge = **$480 ADT**
- **Baccarat**: $100 avg bet × 80 hands/hr × 5 hours × 1.06% edge = **$424 ADT**

### Reinvestment Rates (% of Theoretical Returned as Comps)
| Player Tier | Reinvestment Rate | Typical ADT Range |
|------------|------------------|-------------------|
| Casual | 10-15% | Under $100 |
| Mid-tier | 15-20% | $100-$500 |
| Premium | 20-30% | $500-$2,000 |
| High-roller | 30-40% | $2,000+ |

### Comp Hierarchy (Lowest to Highest Value)
1. **Complimentary beverages** ($2-5 per drink) — virtually all active players
2. **Meal comps** — casual ($10-20), mid-tier ($25-50), premium ($100-500)
3. **Hotel room comps** — discounts 10-20%, free standard, free suite ($1,000-5,000/night)
4. **Show/entertainment tickets** — discounted to VIP seating to private events
5. **Ground transportation** — valet, limo ($50-200 per instance)
6. **Airfare comps** — economy to first class
7. **Private jet service** — ultra-premium players only (tens of thousands per trip)
8. **Loss rebates** — 5-10% of net loss returned
9. **Cashback** — direct currency rebates on theoretical loss

### Comp Approval Hierarchy
| Authority Level | Comp Value | Examples |
|----------------|-----------|---------|
| Pit Supervisor | Under $100 | Meal voucher, small free play |
| Pit Boss/Manager | $100-$500 | Show tickets, room discount |
| Casino Host | $500-$1,000 | Room + dining + entertainment package |
| Marketing/Casino Mgmt | $1,000+ | RFB packages, airfare, loss rebates, private jet |

### Major Loyalty Tier Systems
**Caesars Rewards**: Gold → Platinum (5K TC) → Diamond (15K TC) → Seven Stars → Nobu
**MGM Rewards**: Sapphire → Gold → Platinum (200K TC) → NOIR
**Hard Rock Unity**: Basic → Elevated → Boss → Legend

### Trip-Level vs. Cumulative Theoretical
- **Trip-level**: Drives near-term offers (bounce-back packages, visit-specific comps)
- **Cumulative** (rolling 12-month): Drives tier status, annual benefits, long-term comp eligibility
- Sophisticated casinos use both: trip-level for promotional comps, cumulative for tier status

---

## 3. Casino CRM Data Models

### Player Profile — Core Fields
| Category | Key Fields |
|----------|-----------|
| Identity | PlayerID, FirstName, LastName, DOB, SSN (encrypted), Gender |
| Contact | PrimaryPhone, MobilePhone, Email, PreferredContactMethod, OptInFlags |
| Address | Street, City, State, Zip, Country, AddressType (residential/business/seasonal) |
| Gaming Prefs | PreferredGameTypes, PreferredDenominations, PreferredTableLimits, AvgSessionDuration |
| Hospitality Prefs | PreferredRoomType, FloorPreference, DiningVenues, DietaryRestrictions, AccessibilityNeeds |
| Special Dates | NextBirthday, Anniversary, FirstVisitDate, MilestoneAnniversaries |
| VIP Status | CurrentTier, TierStartDate, TierExpiration, HostAssignmentID, TierPointsEarned |

### Visit/Trip History
| Field | Description |
|-------|------------|
| VisitID | Unique visit identifier |
| VisitStart/End | DateTime stamps |
| TotalWagered | Total amount wagered across all games |
| ActualNetWinLoss | Actual money won/lost |
| TotalTheoreticalWin | Casino's expected profit |
| SlotActivity | MachineID, CoinIn, CoinOut, TimePlayed, Spins |
| TableActivity | GameType, BuyIn, AvgBet, HighBet, HoursPlayed |
| CompUsage | CompType, Value, AuthorizedBy, WasRedeemed |
| HotelStay | CheckIn/Out, RoomType, Rate, IsComp |

### Communication History
| Field | Description |
|-------|------------|
| OutreachType | EMAIL, SMS, PHONE_CALL, POSTAL, PUSH_NOTIFICATION |
| Channel | MARKETING_CAMPAIGN, HOST_PERSONAL, VIP_EVENT_INVITATION |
| CampaignID | Link to parent campaign |
| DeliveryStatus | SUCCESS, BOUNCED, UNDELIVERABLE |
| ResponseType | CALL_RESPONSE, EMAIL_CLICK, VISIT, NO_RESPONSE |
| OfferHistory | OfferID, Value, IsRedeemed, DaysToRedemption |
| PreferredContactTime | HourStart, HourEnd, PreferredDays |
| FrequencyCap | Max contacts per week/month per channel |

### Financial Metrics
| Metric | Description |
|--------|------------|
| LifetimeTheoreticalWin | Cumulative expected casino profit |
| LifetimeActualWinLoss | Cumulative real results |
| CurrentPeriodADT | Rolling 12-month average daily theoretical |
| ADTTrendArray | Quarterly ADT values for trend analysis |
| CompBudget | ADT × CompAllocationPercentage (20-35%) |
| CompBudgetRemaining | Unspent comp allocation |
| ChurnRiskScore | Based on days since last visit vs. baseline frequency |
| EngagementTrendIndicator | Recent quarter ADT vs. prior year |

### Loyalty/Points System
| Field | Description |
|-------|------------|
| CurrentTier | PLATINUM, GOLD, SILVER, BRONZE, STANDARD |
| TierPointsEarned | Running total in qualification period |
| TierPointsRequired | Threshold for current tier |
| PointsLedger | Transaction-level point history (earned, redeemed, expired) |
| RedemptionHistory | What points were redeemed for, when, where |

### Integration Architecture
| System | Integration Type | Data Flow |
|--------|-----------------|-----------|
| PMS (Hotel) | Real-time API | Player profile ↔ Room assignment, check-in events |
| POS (F&B) | Event-driven | Transaction events → Player visit record |
| Ticketing | API query | Tier status → Entertainment credit eligibility |
| Transportation | API query | Tier status → Limo/transport comp eligibility |
| Online Gaming | Real-time sync | Online play data → Unified player profile |
| Marketing Automation | Batch + real-time | Segments → Campaigns → Response tracking |

### Real-Time vs. Batch Processing
| Use Case | Processing Type | Latency |
|----------|----------------|---------|
| Slot machine promotions | Real-time (Kafka/Kinesis) | Milliseconds |
| Player session monitoring | Real-time | Seconds |
| Host mobile app data | Near real-time (ODS) | Seconds to minutes |
| Loyalty points calculation | Hybrid | Real-time display, nightly reconciliation |
| Historical analytics | Batch (nightly) | Hours |
| ML model training | Batch (weekly) | Hours |
| Regulatory reports | Batch | Scheduled |

---

## 4. Key Metrics for Hey Seven

### Player Valuation Metrics
- **ADT** (Average Daily Theoretical): Primary player value metric
- **ADW** (Average Daily Worth): Better of ADT or % of actual loss
- **True Value** (predictive): ML-predicted lifetime value (Gaming Analytics approach)
- **Total Guest Value (TGV)**: Gaming + non-gaming spend combined

### Host Performance Metrics
- Monthly active VIPs maintained
- Average Revenue Per VIP (ARPV)
- Time-to-resolution for player issues
- Trips generated from assigned players
- New player acquisitions
- Player retention rate / churn rate
- Reinvestment efficiency (ROI on comps)

### Key Industry Benchmarks
- Top 5% of players generate 50-80% of gaming revenue
- Typical host portfolio: 300-450 players
- Active contact target: 95% of active players per quarter
- Comp reinvestment: 10-40% of theoretical (tier-dependent)
- ADT threshold for host assignment: $300-500+
- Player reactivation most effective within 3-10 days of churn signal
