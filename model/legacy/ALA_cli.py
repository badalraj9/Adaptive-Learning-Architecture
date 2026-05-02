"""
ALA_cli.py
Adaptive Learning Architecture — Command Line Interface
You talk to her. She responds. She learns. She grows.

Commands:
    just type anything     → ALA perceives and responds
    /teach <text>          → teach ALA something
    /status                → see ALA's internal state
    /save                  → save ALA's state
    /feeling               → how is ALA feeling right now
    /primitives            → list formed primitives
    /timeline              → show ALA's life snapshots
    /help                  → show commands
    /quit                  → save and exit

Run in Colab:
    !python ALA_cli.py --demo              (scripted demo)
    !python ALA_cli.py --demo --no-companion  (faster)

Run on PC:
    python ALA_cli.py                      (interactive)
    python ALA_cli.py --no-companion       (faster)
"""

import sys
import os
import argparse
sys.path.insert(0, '/content')

from ALA_full import ALA


# ─────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"


def feeling_color(feeling):
    return {
        "settled":   C.GREEN,
        "uncertain": C.YELLOW,
        "confused":  C.RED
    }.get(feeling, C.WHITE)

def feeling_symbol(feeling):
    return {
        "settled":   "●",
        "uncertain": "◐",
        "confused":  "○"
    }.get(feeling, "?")


# ─────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────

def print_banner():
    print(f"\n{C.CYAN}{C.BOLD}")
    print("╔══════════════════════════════════════╗")
    print("║         ALA — Adaptive Learning      ║")
    print("║         Architecture v7.1            ║")
    print("║                                      ║")
    print("║  Not a model you run.                ║")
    print("║  A system you live with.             ║")
    print("╚══════════════════════════════════════╝")
    print(f"{C.RESET}")

def print_help():
    print(f"\n{C.DIM}Commands:{C.RESET}")
    print(f"  {C.CYAN}/teach <text>{C.RESET}   → teach ALA something")
    print(f"  {C.CYAN}/status{C.RESET}         → ALA's internal state")
    print(f"  {C.CYAN}/save{C.RESET}           → save ALA's state")
    print(f"  {C.CYAN}/feeling{C.RESET}        → how is ALA feeling")
    print(f"  {C.CYAN}/primitives{C.RESET}     → formed concepts")
    print(f"  {C.CYAN}/timeline{C.RESET}       → ALA's life snapshots")
    print(f"  {C.CYAN}/help{C.RESET}           → this list")
    print(f"  {C.CYAN}/quit{C.RESET}           → save and exit")
    print(f"\n  {C.DIM}anything else → ALA perceives and responds{C.RESET}\n")

def print_state_bar(state):
    feeling = state["feeling"]
    color   = feeling_color(feeling)
    symbol  = feeling_symbol(feeling)
    q       = state["settling_quality"]
    s       = state["salience"]
    n       = state["n_primitives"]
    step    = state["step"]

    extras = ""
    if state["companion_queried"]:
        extras += f" {C.YELLOW}[asked companion]{C.RESET}{C.DIM}"
    if state["primitive_formed"] is not None:
        extras += f" {C.GREEN}[concept formed!]{C.RESET}{C.DIM}"

    print(f"{C.DIM}  {color}{symbol}{C.RESET}{C.DIM} "
          f"feeling:{feeling} quality:{q:.2f} "
          f"salience:{s:.2f} concepts:{n} "
          f"step:{step}{extras}{C.RESET}")

def print_ala_response(response, state):
    feeling = state["feeling"]
    color   = feeling_color(feeling)
    print(f"\n{color}{C.BOLD}ALA:{C.RESET} {response}")
    print_state_bar(state)

def print_primitives(ala):
    n        = ala.slow.store.n()
    archived = len(ala.slow.store.archive)
    density  = ala.slow.assoc.density()

    print(f"\n{C.CYAN}Concepts (primitives):{C.RESET}")
    print(f"  Active:          {n}")
    print(f"  Archived:        {archived}")
    print(f"  W_assoc density: {density:.4f}")

    if n > 0:
        print(f"\n  Active concept norms:")
        for i, p in enumerate(ala.slow.store.primitives[:5]):
            usage = ala.slow.store.usage_count[i]
            print(f"    [{i}] norm={p.norm().item():.3f} usage={usage}")
        if n > 5:
            print(f"    ... and {n-5} more")

def print_timeline(ala):
    snaps = ala.manager.list_timeline()
    print(f"\n{C.CYAN}Timeline — ALA's life:{C.RESET}")
    if not snaps:
        print(f"  {C.DIM}No snapshots yet{C.RESET}")
    for snap in snaps:
        print(f"  📸 {snap}")


# ─────────────────────────────────────────
# TEACHING MODE
# ─────────────────────────────────────────

def teaching_mode(ala, content):
    words  = content.split()
    before = ala.slow.store.n()

    print(f"{C.DIM}Teaching '{content[:50]}...' "
          f"({len(words)} words){C.RESET}")

    for word in words:
        _, state = ala.perceive(word, "text")
        if state["primitive_formed"] is not None:
            print(f"{C.GREEN}  ★ New concept formed at step {state['step']}!{C.RESET}")

    after = ala.slow.store.n()
    new   = after - before
    print(f"{C.DIM}Done. Concepts: {before} → {after}"
          f"{(' (+' + str(new) + ')') if new > 0 else ''}{C.RESET}")


# ─────────────────────────────────────────
# AUTOSAVE
# ─────────────────────────────────────────

AUTOSAVE_EVERY = 50

def maybe_autosave(ala):
    if ala.step_count % AUTOSAVE_EVERY == 0 and ala.step_count > 0:
        ala.save(notes=f"autosave step {ala.step_count}")
        print(f"{C.DIM}  [autosaved]{C.RESET}")


# ─────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────

def run_cli(state_dir="ALA_state", use_companion=True):
    print_banner()
    print(f"{C.DIM}Loading ALA...{C.RESET}")

    ala = ALA(
        state_dir     = state_dir,
        use_companion = use_companion,
        verbose       = False
    )

    ala.status()
    print_help()
    print(f"{C.CYAN}Ready. Start talking to ALA.{C.RESET}\n")

    while True:
        try:
            user_input = input(f"{C.BOLD}{C.WHITE}You: {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Interrupted.{C.RESET}")
            break

        if not user_input:
            continue

        if user_input.startswith("/teach "):
            content = user_input[7:].strip()
            if content:
                teaching_mode(ala, content)
            else:
                print(f"{C.DIM}Usage: /teach <text>{C.RESET}")

        elif user_input == "/status":
            ala.status()

        elif user_input == "/save":
            ala.save(notes="manual save")
            print(f"{C.GREEN}Saved.{C.RESET}")

        elif user_input == "/feeling":
            n       = ala.slow.store.n()
            density = ala.slow.assoc.density()
            print(f"\n{C.CYAN}ALA is feeling:{C.RESET}")
            print(f"  Concepts:    {n}")
            print(f"  Connections: {density:.4f}")
            print(f"  Steps lived: {ala.step_count}")
            if n == 0:
                print(f"  {C.RED}Newborn. Everything is unfamiliar.{C.RESET}")
            elif n < 10:
                print(f"  {C.YELLOW}Beginning to recognize patterns.{C.RESET}")
            else:
                print(f"  {C.GREEN}Building understanding.{C.RESET}")

        elif user_input == "/primitives":
            print_primitives(ala)

        elif user_input == "/timeline":
            print_timeline(ala)

        elif user_input == "/help":
            print_help()

        elif user_input == "/quit":
            print(f"\n{C.DIM}Saving...{C.RESET}")
            ala.save(notes="session end")
            print(f"{C.GREEN}ALA saved. She will resume next time.{C.RESET}")
            print(f"{C.DIM}Goodbye.{C.RESET}\n")
            break

        else:
            response, state = ala.perceive(user_input, "text")
            print_ala_response(response, state)
            maybe_autosave(ala)


# ─────────────────────────────────────────
# COLAB DEMO — scripted, non-interactive
# ─────────────────────────────────────────

def run_colab_demo(state_dir="ALA_state", use_companion=False):
    print_banner()
    print(f"{C.DIM}Colab demo mode (non-interactive){C.RESET}\n")

    ala = ALA(
        state_dir     = state_dir,
        use_companion = use_companion,
        verbose       = False
    )

    ala.status()

    print(f"\n{C.CYAN}{'─'*40}{C.RESET}")
    print(f"{C.CYAN}  Demo Conversation{C.RESET}")
    print(f"{C.CYAN}{'─'*40}{C.RESET}\n")

    # first — teach her a lot
    # more repetition = more primitive formation
    print(f"{C.CYAN}Phase 1: Teaching{C.RESET}\n")

    lessons = [
        "cat dog bird fish animal",
        "cat is an animal dog is an animal",
        "bird is an animal fish is an animal",
        "cat meows dog barks bird sings",
        "animals breathe animals move animals eat",
        "cat dog bird fish are animals",
        "all animals need food all animals breathe",
        "cat is a small animal dog is a loyal animal",
        "bird can fly fish can swim cat can run dog can run",
        "animals are living things animals grow animals feel",
    ]

    for lesson in lessons:
        teaching_mode(ala, lesson)

    print(f"\n{C.CYAN}Phase 2: Conversation{C.RESET}\n")

    questions = [
        "hello",
        "what is a cat",
        "tell me about animals",
        "do animals breathe",
        "what can a bird do",
        "is a fish an animal",
        "what do animals need",
        "are you learning",
    ]

    for q in questions:
        print(f"{C.BOLD}{C.WHITE}You: {C.RESET}{q}")
        response, state = ala.perceive(q, "text")
        print_ala_response(response, state)
        print()

    print(f"\n{C.CYAN}{'─'*40}{C.RESET}")
    print(f"{C.CYAN}  After Demo{C.RESET}")
    print(f"{C.CYAN}{'─'*40}{C.RESET}")
    ala.status()
    print_primitives(ala)
    print_timeline(ala)

    ala.save(notes="colab demo")
    print(f"\n{C.GREEN}ALA saved.{C.RESET}")
    print(f"{C.DIM}She resumes next time you run this.{C.RESET}")
    print(f"\n{C.CYAN}To keep teaching her:{C.RESET}")
    print(f"  !python ALA_cli.py --demo")
    print(f"\n{C.CYAN}On your PC (interactive):{C.RESET}")
    print(f"  python ALA_cli.py\n")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talk to ALA")
    parser.add_argument(
        "--state", default="ALA_state",
        help="State directory (default: ALA_state)"
    )
    parser.add_argument(
        "--no-companion", action="store_true",
        help="Skip companion (faster)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Colab demo mode"
    )
    args = parser.parse_args()

    use_companion = not args.no_companion
    in_colab      = os.path.exists('/content')

    if args.demo or in_colab:
        run_colab_demo(
            state_dir     = args.state,
            use_companion = use_companion
        )
    else:
        run_cli(
            state_dir     = args.state,
            use_companion = use_companion
        )
