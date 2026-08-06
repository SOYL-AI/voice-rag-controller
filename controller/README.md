# controller/ — The Decision Brain

**Owner: Renya**

## What this folder is

The controller is the core research idea of the whole project: one small component that 
decides, during a conversation, when to look something up, what to remember, and when to 
speak. In Phase 1 it does nothing yet — it is a no-op stub that just sits in the loop and 
logs that it was asked to decide.

## Why it exists now (even as a stub)

We put the controller in the loop from Day 1 so that later, when we add real decision logic, 
we are dropping it into a slot that already exists, rather than scrambling to insert it. Its 
interface is deliberately simple: it receives a snapshot of the current situation and 
returns an action.

## Phase 1 goal (deliverable)

A stub function, decide(belief) -> action, that:
1. Takes whatever situation snapshot the pipeline gives it.
2. Always returns a "do nothing" action for now.
3. Logs a controller_action event with action "noop" in the frozen schema.

That is all for Phase 1. The point is that the slot exists and is logged.

## Background: what it becomes later

In later phases this is where the real logic lives: reading the model's uncertainty, the 
timing signals, and the memory state, and deciding whether to retrieve, wait, speak, or 
summarize, based on whether an action is worth the delay it causes. The interface stays the 
same (decide(belief) -> action), so the real logic is a drop-in replacement for this stub.

---

## Phase 2 — First Real Decision Logic

In Phase 1 the controller was a no-op stub that always returned "do nothing." In Phase 2 it 
gets its first real rule, so it actually starts making decisions during a conversation.

### Goal

Replace the no-op stub with a simple, readable rule that decides whether to retrieve, based 
on the model's uncertainty. Keep the same decide(belief) interface so it stays a clean 
drop-in.

### How to do it

- The controller receives a "belief" snapshot each turn, which now includes real signals 
  (the model's confidence numbers from Ryan's instrumentation, the retrieval scores, the 
  tracker's uncertainty).
- Start with one simple hand-written rule, for example: if the model's confidence on what 
  it is about to say is below a threshold, decide to retrieve; otherwise, do not. Put the 
  threshold in configs, not hardcoded.
- Log the decision as a controller_action event, with the action taken (retrieve, wait, 
  speak) instead of always noop.
- Keep the logic in a plain rules file, readable and easy to change, since the whole point 
  is that we can swap and tune the rule for experiments.

### Definition of done

The controller reads the real belief signals and makes a genuine retrieve-or-not decision 
each turn, logged correctly, replacing the no-op. The interface (decide(belief) -> action) 
stays the same.

### Note

This simple rule is the baseline. Later, more sophisticated decision logic (and the 
experiments comparing approaches) build on top of this same interface. The value comes from 
keeping the rule simple and interpretable first.

### Coordinate with

Ryan (whose signal instrumentation provides the confidence numbers the rule reads) and the 
pipeline integration (this rule plugs into the integrated spine).
