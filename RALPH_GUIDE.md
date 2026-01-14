# Ralph Workflow Guide

Based on [The Ralph Playbook](https://github.com/ghuntley/how-to-ralph-wiggum) by Geoffrey Huntley and Clayton Farr.

---

## Overview: Three Phases, Two Prompts, One Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PLANNING PHASE                                │
│                                                                      │
│  ┌──────────────────────┐      ┌──────────────────────┐             │
│  │  REQUIREMENTS PHASE  │      │     TODO PHASE       │             │
│  │  (Interactive)       │      │  (./loop.sh plan)    │             │
│  │                      │      │                      │             │
│  │  Human + Claude      │─────▶│  Gap analysis        │             │
│  │  conversation        │      │  specs vs code       │             │
│  │                      │      │                      │             │
│  │  OUTPUT: specs/*.md  │      │  OUTPUT:             │             │
│  │  (one per topic)     │      │  IMPLEMENTATION_     │             │
│  └──────────────────────┘      │  PLAN.md             │             │
│                                └──────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     IMPLEMENTATION PHASE                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    INCREMENTAL LOOP                            │ │
│  │                    (./loop.sh)                                 │ │
│  │                                                                │ │
│  │   while true; do cat PROMPT_build.md | claude -p ... ; done   │ │
│  │                                                                │ │
│  │   Each iteration = fresh context = one task = one commit      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1A: Requirements Phase (This Document)

### What Is It?

An **interactive conversation** between you (human) and Claude to define WHAT to build. This is NOT automated - it's a collaborative discussion.

### Key Concepts

#### Jobs to Be Done (JTBD)
A high-level user need or outcome. Example:
> "Help quantitative traders backtest strategies with deep learning integration"

#### Topics of Concern
Distinct aspects/components within a JTBD. Each topic becomes ONE spec file.

#### The "One Sentence Without 'And'" Test

**Critical rule**: If you need "and" to describe what a topic does, it's probably multiple topics.

| ✓ Good (single topic) | ✗ Bad (multiple topics) |
|-----------------------|-------------------------|
| "The data module loads market data from various formats" | "The system handles data loading, backtesting, and reporting" |
| "The metrics module calculates risk-adjusted returns" | "The engine manages positions and calculates performance" |

#### Relationships

```
1 JTBD
  └── multiple topics of concern
        └── 1 topic = 1 spec file
              └── 1 spec = multiple tasks (in IMPLEMENTATION_PLAN.md)
```

---

## Working with Existing Codebases (Brownfield Projects)

**Critical insight**: Specs are ASPIRATIONAL, not DESCRIPTIVE.

### The Wrong Approach

❌ "Let me read the code and document what exists as specs"

This is backwards. You're not writing documentation - you're defining requirements.

### The Right Approach

✅ "Let me define what SHOULD exist, then let gap analysis find what's missing"

```
┌─────────────────────────────────────────────────────────────────────┐
│  YOU define specs for ideal system (what SHOULD exist)              │
│                           │                                         │
│                           ▼                                         │
│  ./loop.sh plan runs gap analysis:                                  │
│    • Studies existing src/ with subagents                           │
│    • Compares specs against actual code                             │
│    • Identifies TRUE gaps (not already implemented)                 │
│                           │                                         │
│                           ▼                                         │
│  IMPLEMENTATION_PLAN.md contains only MISSING items                 │
│                           │                                         │
│                           ▼                                         │
│  ./loop.sh builds only what's needed                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Principles for Existing Code

1. **Don't reverse-engineer specs from code**
   - Specs define the ideal state
   - Gap analysis discovers current state
   - The delta becomes the implementation plan

2. **Your existing code steers Ralph**
   - Existing patterns in `src/` influence what Ralph generates
   - If Ralph produces wrong patterns, add utilities/examples to steer it

3. **The planning phase handles reconciliation**
   - `PROMPT_plan.md` includes: "Do NOT assume functionality is missing; confirm with code search first"
   - Ralph uses up to 500 subagents to study existing code
   - Only true gaps make it to the implementation plan

4. **Specs can be ambitious**
   - Include features you WANT, even if not yet implemented
   - Gap analysis will correctly identify them as TODO items

### Example: Existing Backtest Engine

You have code in `src/` that already handles data loading and basic backtesting.

**Don't**: Write specs that only describe what exists
**Do**: Write specs for the COMPLETE system you want

```
specs/data-handling.md:
  - CSV loading ✓ (exists)
  - Parquet loading ✓ (exists)
  - Real-time streaming (aspirational - not yet built)
  - Corporate actions (aspirational - not yet built)
```

Gap analysis will find that streaming and corporate actions are missing, and add them to `IMPLEMENTATION_PLAN.md`.

---

## How to Execute the Requirements Phase

### Step 1: Define the JTBD

Start with a clear statement of what outcome we're trying to achieve:

```
JTBD: ________________________________________________
```

### Step 2: Identify Topics of Concern

Break the JTBD into distinct topics. For each proposed topic, apply the test:

> "Can I describe this in ONE sentence WITHOUT 'and'?"

List your topics:

```
Topic 1: ______________ → "The ___ module _______________"
Topic 2: ______________ → "The ___ module _______________"
Topic 3: ______________ → "The ___ module _______________"
...
```

### Step 3: For Each Topic, Discuss and Document

Have a conversation about each topic covering:

1. **Purpose** - What does it do? (one sentence)
2. **Requirements** - What must it support?
3. **Acceptance Criteria** - How do we know it's done?
4. **Edge Cases** - What could go wrong?
5. **Dependencies** - What does it need from other topics?

### Step 4: Claude Writes the Spec

After discussion, Claude writes `specs/TOPIC_NAME.md` with:

```markdown
# Topic Name

## Purpose
One sentence description.

## Requirements
- Requirement 1
- Requirement 2
- ...

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- ...

## Edge Cases
- Edge case 1: How to handle
- Edge case 2: How to handle

## Dependencies
- Depends on: X, Y
- Depended on by: Z
```

---

## Example: Backtest Engine Topics

For a Rust backtest engine with DL integration, the topics might be:

| Topic | One-Sentence Description | Spec File |
|-------|--------------------------|-----------|
| Core Engine | "The core engine executes strategies against historical price data" | `specs/core-engine.md` |
| Data Handling | "The data module loads and normalizes market data from various formats" | `specs/data-handling.md` |
| Position Management | "The position manager tracks holdings, P&L, and portfolio state" | `specs/positions.md` |
| Performance Metrics | "The metrics module calculates risk-adjusted performance statistics" | `specs/metrics.md` |
| ML Integration | "The ML module exports data and integrates signals for deep learning workflows" | `specs/ml-integration.md` |
| CLI Interface | "The CLI provides command-line access to backtest functionality" | `specs/cli.md` |

---

## After Requirements Phase

Once all specs are written:

```bash
# Verify specs exist
ls specs/

# Run TODO phase to generate implementation plan
./loop.sh plan 1

# Review the generated IMPLEMENTATION_PLAN.md
cat IMPLEMENTATION_PLAN.md

# If plan looks wrong, regenerate
./loop.sh plan 1

# When plan is good, start building
./loop.sh
```

---

## Tips

### Let Claude Use Subagents for Research

During the conversation, Claude can spawn subagents to:
- Read documentation URLs
- Study existing code patterns
- Research best practices

Just ask: "Can you research how other Rust backtesting frameworks handle X?"

### Specs Are Living Documents

If during building Ralph discovers inconsistencies in specs, it will update them (per PROMPT_build.md guardrail). This is expected.

### The Plan Is Disposable

If the IMPLEMENTATION_PLAN.md goes wrong:
1. Delete it
2. Run `./loop.sh plan 1` again
3. Cost: one planning iteration (cheap)

### Don't Over-Specify

Specs define WHAT and WHY, not HOW. Let Ralph decide implementation details during the build phase.

---

## Checklist Before Moving to TODO Phase

- [ ] JTBD is clearly defined
- [ ] All topics pass the "one sentence without and" test
- [ ] Each topic has its own `specs/TOPIC.md` file
- [ ] Each spec has: Purpose, Requirements, Acceptance Criteria
- [ ] No topic is trying to do too much
- [ ] Dependencies between topics are noted

---

## Ready?

Once this checklist is complete, proceed to:

```bash
./loop.sh plan
```
