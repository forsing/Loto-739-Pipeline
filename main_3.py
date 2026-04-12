# -*- coding: utf-8 -*-



# Inspiration - Inspiracija
## https://github.com/Callam7/LottoPipeline



"""
cd /Users/4c/Desktop/GHQ/kurzor/LottoPipeline-main
python3 main_3.py
"""



import os
import warnings
import logging
import random
import numpy as np
from datetime import datetime

# Kei startup clean: suppress known non-critical runtime warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=Warning, module="pennylane")
warnings.filterwarnings(
    "ignore",
    message=r"At this time, the v2\.11\+ optimizer `tf\.keras\.optimizers\.Adam` runs slowly on M1/M2 Macs.*",
)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# Database functions
from database import (
    initialize_database,
    fetch_recent_draws,
    fetch_all_draws,
    insert_draw
)

# Data I/O
from data_io import load_current_ticket, save_current_ticket
from data_io import CURRENT_TICKET_FILE

# Pipeline structure
from pipeline import DataPipeline
from steps.generate_ticket import generate_ticket


# Pipeline steps
from steps.historical import process_historical_data
from steps.frequency import analyze_number_frequency
from steps.decay import calculate_decay_factors
from steps.clustering import kmeans_clustering_and_correlation
from steps.monte_carlo import monte_carlo_simulation
from steps.redundancy import sequential_features
from steps.markov import markov_features
from steps.entropy import shannon_entropy_features
from steps.bayesian_fusion import bayesian_fusion_with_mechanics
# Deep learning is imported lazily inside option 3 so the app can run
# even when TensorFlow/NumPy binaries are incompatible in the environment.

# Constants
NUM_PICK_MAIN = 7
MAX_MAIN_NUMBER = 39
NUM_POWERBALL = 1
TICKET_LINES = 1
RANDOM_SEED = 39


# ============================================================
# Utility Functions
# ============================================================
def _normalize_line_to_7(numbers, fallback=None):
    """
    Normalize a ticket line to exactly 7 unique numbers in range 1..39.
    Used to migrate older 6-number saved tickets.
    """
    vals = []
    for n in numbers or []:
        try:
            n = int(n)
        except Exception:
            continue
        if 1 <= n <= 39 and n not in vals:
            vals.append(n)
    if fallback is not None:
        try:
            f = int(fallback)
            if 1 <= f <= 39 and f not in vals:
                vals.append(f)
        except Exception:
            pass
    # Deterministic fill if still short
    for n in range(1, 40):
        if len(vals) >= 7:
            break
        if n not in vals:
            vals.append(n)
    return sorted(vals[:7])


def verify_draw_order():
    """Verify that draw_id reflects chronological order."""
    all_draws = fetch_all_draws()
    if not all_draws:
        return
    dates = [draw['draw_date'] for draw in all_draws]
    if dates == sorted(dates):
        print("Verification Passed: draw_id correctly reflects chronological order.")
    else:
        print("Verification Failed: draw_id does NOT correctly reflect chronological order.")


def get_latest_draw_date():
    """Return the most recent draw date."""
    all_draws = fetch_all_draws()
    if not all_draws:
        return None
    latest_draw = all_draws[-1]
    try:
        return datetime.strptime(latest_draw['draw_date'], "%Y-%m-%d")
    except ValueError:
        return None


def view_number_stats(pipeline):
    """Display frequency stats for 7/39 main numbers and placeholder PB."""
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        print("No historical data in pipeline. Fetching from database...")
        all_draws = fetch_all_draws()
        if not all_draws:
            print("No draws in database.")
            return
        pipeline.add_data("historical_data", all_draws)
        historical_data = all_draws

    number_frequency = pipeline.get_data("number_frequency")
    powerball_frequency = pipeline.get_data("powerball_frequency")

    if number_frequency is None or powerball_frequency is None:
        print("Frequency data missing. Running analysis...")
        analyze_number_frequency(pipeline)
        number_frequency = pipeline.get_data("number_frequency")
        powerball_frequency = pipeline.get_data("powerball_frequency")

    print("\n--- Main Numbers Frequency (1..39) ---")
    print("Number | Occurrences | % of main picks")
    total_main_picks = len(historical_data) * NUM_PICK_MAIN
    for i in range(MAX_MAIN_NUMBER):
        count = int(number_frequency[i] * total_main_picks)
        percent = number_frequency[i] * 100
        print(f"{i+1:2d}     | {count:10d}   | {percent:6.2f}%")

    print("\n--- Powerball Placeholder Frequency (fixed=1) ---")
    print("Number | Occurrences | % of Powerball picks")
    total_powerball_picks = len(historical_data)
    for i in range(NUM_POWERBALL):
        count = int(powerball_frequency[i] * total_powerball_picks)
        percent = powerball_frequency[i] * 100
        print(f"{i+1:2d}     | {count:10d}   | {percent:6.2f}%")


# ============================================================
# Safe Execution Wrapper
# ============================================================
def safe_run(step_fn, pipeline, name):
    """Safely execute pipeline stage with error handling."""
    try:
        step_fn(pipeline)
        print(f"[OK] {name} completed.")
    except Exception as e:
        print(f"[ERROR] {name} failed: {e}")


def _reset_determinism_per_run(seed=RANDOM_SEED):
    """
    Reset RNG state before each NEXT computation so repeated runs
    with unchanged CSV produce the same combination.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    # Reset quantum encoder global weights to deterministic initial state
    try:
        import config.quantum_features as qf
        qf.np.random.seed(seed)
        qf._global_weights = qf.np.random.normal(
            loc=0.0,
            scale=0.1,
            size=(qf._Q_NUM_LAYERS, qf.NUM_QUBITS, 3),
        )
    except Exception:
        pass


def run_next_combination(pipeline, all_draws):
    """
    Execute full feature pipeline and return one NEXT combination.
    Used by option 1 (refresh from CSV/DB) and option 3.
    """
    _reset_determinism_per_run()
    pipeline.clear_pipeline()
    pipeline.add_data("historical_data", all_draws)

    safe_run(lambda p: process_historical_data({"past_results": all_draws}, p), pipeline, "Historical Processing")
    safe_run(analyze_number_frequency, pipeline, "Frequency Analysis")
    safe_run(calculate_decay_factors, pipeline, "Decay Calculation")
    safe_run(bayesian_fusion_with_mechanics, pipeline, "Bayesian Fusion")
    safe_run(kmeans_clustering_and_correlation, pipeline, "Clustering")
    safe_run(monte_carlo_simulation, pipeline, "Monte Carlo Simulation")
    safe_run(sequential_features, pipeline, "Sequential/Redundancy")
    safe_run(markov_features, pipeline, "Markov Features")
    safe_run(shannon_entropy_features, pipeline, "Entropy Features")

    dl_enabled = True
    try:
        from steps.deep_learning import deep_learning_prediction
        safe_run(deep_learning_prediction, pipeline, "Deep Learning Prediction")
    except Exception as e:
        dl_enabled = False
        print(f"[WARN] Deep Learning step skipped (env issue): {e}")
    print(f"[INFO] Deep Learning status: {'ON' if dl_enabled else 'OFF'}")

    new_ticket = generate_ticket(pipeline)
    return new_ticket


# ============================================================
# Main Program Loop
# ============================================================
def main():
    initialize_database()
    verify_draw_order()
    pipeline = DataPipeline()

    while True:
        print("\n--- Lotto Predictor Menu ---")
        print("1. Display Current Ticket")
        print("2. List Last 10 Results (from DB)")
        print("3. Insert New Draw & Generate Ticket")
        print("4. Number Stats")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            print("\n[INFO] Option 1 selected: Refresh NEXT Combination (with DL)")
            all_draws = fetch_all_draws()
            if not all_draws:
                print("No historical draws in database/CSV.")
                continue
            new_ticket = run_next_combination(pipeline, all_draws)
            # Keep file strict: one line with 7 numbers.
            if new_ticket:
                line = new_ticket[0]
                migrated_line = _normalize_line_to_7(
                    line.get("line", []),
                    line.get("powerball"),
                )
                save_current_ticket([{"line": migrated_line, "powerball": 1}])
            print("\n--- NEXT Combination ---")
            for idx, line in enumerate(new_ticket, 1):
                print(f"Line {idx}: {line['line']}")

        elif choice == "2":
            print("\n[INFO] Option 2 selected: Last 10 Results")
            last_draws = fetch_recent_draws(10)
            if not last_draws:
                print("No historical draws.")
            else:
                print(f"[INFO] Loaded {len(last_draws)} draw(s).")
                print("\n--- Last 10 Draws (CSV-synced DB) ---")
                for draw in last_draws:
                    print(f"Date: {draw['draw_date']} | Numbers: {draw['numbers']} | "
                          f"Bonus(7th): {draw['bonus']}")

        elif choice == "3":
            draw_date = input("Enter draw date (YYYY-MM-DD) or press Enter for today: ")
            if not draw_date:
                draw_date = datetime.now().strftime("%Y-%m-%d")
            else:
                try:
                    datetime.strptime(draw_date, "%Y-%m-%d")
                except ValueError:
                    print("Invalid date format.")
                    continue

            latest_date = get_latest_draw_date()
            if latest_date and datetime.strptime(draw_date, "%Y-%m-%d") <= latest_date:
                print(f"Error: New draw date must be after {latest_date.strftime('%Y-%m-%d')}.")
                continue

            try:
                numbers = list(map(int, input(f"Enter {NUM_PICK_MAIN} main numbers (1-{MAX_MAIN_NUMBER}): ").split()))
                if len(numbers) != NUM_PICK_MAIN or len(set(numbers)) != NUM_PICK_MAIN:
                    raise ValueError(f"Exactly {NUM_PICK_MAIN} distinct numbers required.")
                if any(n < 1 or n > MAX_MAIN_NUMBER for n in numbers):
                    raise ValueError(f"Numbers must be between 1 and {MAX_MAIN_NUMBER}.")
                bonus = int(input(f"Enter 7th number (1-{MAX_MAIN_NUMBER}): "))
                if not (1 <= bonus <= MAX_MAIN_NUMBER):
                    raise ValueError(f"7th number must be between 1 and {MAX_MAIN_NUMBER}.")
                powerball = int(input(f"Enter placeholder value (must be {NUM_POWERBALL}): "))
                if not (1 <= powerball <= NUM_POWERBALL):
                    raise ValueError(f"Placeholder must be {NUM_POWERBALL}.")
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue

            new_id = insert_draw(draw_date, numbers, bonus, powerball)
            if not new_id:
                print("Error inserting draw.")
                continue
            print(f"New draw inserted with draw_id = {new_id}")

            all_draws = fetch_all_draws()
            new_ticket = run_next_combination(pipeline, all_draws)
            print("\nNEXT combination:")
            for idx, line in enumerate(new_ticket, 1):
                print(f"Line {idx}: {line['line']}")

        elif choice == "4":
            view_number_stats(pipeline)

        elif choice == "5":
            print("Exiting.")
            break

        else:
            print("Invalid choice. Select 1-5.")


if __name__ == "__main__":
    main()





"""
Verification Passed: draw_id correctly reflects chronological order.
2026-04-12 20:18:18,030 - INFO - Initialized DataPipeline.

--- Lotto Predictor Menu ---
1. Display Current Ticket
2. List Last 10 Results (from DB)
3. Insert New Draw & Generate Ticket
4. Number Stats
5. Exit
Enter your choice (1-5): 1

[INFO] Option 1 selected: Refresh NEXT Combination (with DL)
2026-04-12 20:18:41,325 - INFO - Pipeline cleared.
Processed 4596 valid historical draws into the pipeline.
[OK] Historical Processing completed.
2026-04-12 20:18:41,329 - INFO - Number frequency analysis completed.
[OK] Frequency Analysis completed.
2026-04-12 20:18:41,348 - INFO - Decay-weighted frequency calculation completed.
[OK] Decay Calculation completed.
2026-04-12 20:18:41,353 - INFO - Bayesian fusion stored successfully
[OK] Bayesian Fusion completed.
2026-04-12 20:18:41,661 - ERROR - Powerball clustering failed: n_samples=1 should be >= n_clusters=2.
2026-04-12 20:18:41,661 - INFO - K-Means clustering completed.
[OK] Clustering completed.
2026-04-12 20:18:41,751 - INFO - Monte Carlo simulation completed with 344700 simulations.
[OK] Monte Carlo Simulation completed.
2026-04-12 20:18:41,762 - INFO - Sequential / Temporal features generated successfully (cluster-modulated).
[OK] Sequential/Redundancy completed.
2026-04-12 20:18:41,827 - INFO - Markov features integrated successfully.
[OK] Markov Features completed.
2026-04-12 20:18:41,827 - INFO - Shannon entropy features generated successfully.
[OK] Entropy Features completed.
2026-04-12 20:19:15,919 - INFO - Quantum encoder training complete.
...
Epoch 4595/4596
  1/123 [..............................] - ETA: 0s - loss: 0.7081 36/123 [=======>......................] - ETA: 0s - loss: 1.0023 75/123 [=================>............] - ETA: 0s - loss: 1.0165113/123 [==========================>...] - ETA: 0s - loss: 1.0221123/123 [==============================] - 0s 2ms/step - loss: 1.0228 - auc: 0.5538 - bin_acc: 0.7568 - mae: 0.4594 - val_loss: 3.1735 - val_auc: 0.4917 - val_bin_acc: 0.6390 - val_mae: 0.3668 - lr: 5.0000e-06
Epoch 4596/4596
  1/123 [..............................] - ETA: 0s - loss: 0.7054 36/123 [=======>......................] - ETA: 0s - loss: 1.0040 76/123 [=================>............] - ETA: 0s - loss: 1.0179115/123 [===========================>..] - ETA: 0s - loss: 1.0229123/123 [==============================] - 0s 2ms/step - loss: 1.0235 - auc: 0.5526 - bin_acc: 0.7578 - mae: 0.4595 - val_loss: 6.9641 - val_auc: 0.4913 - val_bin_acc: 0.5904 - val_mae: 0.3935 - lr: 5.0000e-06
[OK] Deep Learning Prediction completed.
[INFO] Deep Learning status: ON
2026-04-12 20:44:25,451 - INFO - Successfully saved 1 ticket line(s) to 'current_ticket.json'.
2026-04-12 20:44:25,452 - INFO - Successfully saved 1 ticket line(s) to 'current_ticket.json'.

--- NEXT Combination ---
Line 1: [8, 11, 14, 16, 19, 21, 27]




--- Main Numbers Frequency (1..39) ---
Number | Occurrences | % of main picks
 1     |        777   |   2.42%
 2     |        818   |   2.54%
 3     |        821   |   2.55%
 4     |        805   |   2.50%
 5     |        822   |   2.56%
 6     |        809   |   2.51%
 7     |        836   |   2.60%
 8     |        904   |   2.81%
 9     |        836   |   2.60%
10     |          x   |   2.xx%
11     |        853   |   2.65%
12     |        806   |   2.51%
13     |        824   |   2.56%
14     |        800   |   2.49%
15     |          y   |   2.yy%
16     |        828   |   2.57%
17     |        757   |   2.35%
18     |        815   |   2.53%
19     |        805   |   2.50%
20     |          z   |   2.zz%
21     |        825   |   2.56%
22     |        846   |   2.63%
23     |        901   |   2.80%
24     |        830   |   2.58%
25     |        835   |   2.60%
26     |        866   |   2.69%
27     |        782   |   2.43%
28     |        815   |   2.53%
29     |        842   |   2.62%
30     |        781   |   2.43%
31     |        823   |   2.56%
32     |        852   |   2.65%
33     |        846   |   2.63%
34     |        864   |   2.69%
35     |        839   |   2.61%
36     |        783   |   2.43%
37     |        856   |   2.66%
38     |        832   |   2.59%
39     |        844   |   2.62%





rez
Line 1: [3, 12, 15, 21, 24, 27, 32]
"""



"""
CLI meni (while True + input) koji upravlja SQLite (preko database) + DataPipeline objektom. Opcije: prikaži/osveži tiket (generiši NEXT), poslednjih 10 izvlacenja, unos novog kola u bazu pa ponovo generisanje, statistika frekvencija, izlaz. Srce je run_next_combination() — niz koraka pipeline-a pa generate_ticket().

Tehnike / metode po blokovima
Deo	Šta radi
Startup (linije 24-32)
TF_CPP_MIN_LOG_LEVEL, filtriranje upozorenja (PennyLane, Keras na M1/M2), utišavanje TensorFlow/absl — čistiji terminal.
Importi
database (init, fetch, insert), data_io (tiket fajl), pipeline.DataPipeline, generate_ticket, zatim step moduli: istorija, frekvencija, decay, k-means+korelacija, Monte Carlo, sekvenca/redundansa, Markov, entropija, bayesijska fuzija; DL se uvozi lenjo u run_next_combination da skripta radi i bez kompatibilnog TF-a.
Konstante
Loto 7/39 + „powerball“ kao placeholder (1 vrednost), TICKET_LINES = 1, fiksni RANDOM_SEED = 39.
_normalize_line_to_7()
Migracija starijih tiketa sa 6 brojeva na tačno 7 jedinstvenih 1–39; opciono fallback; determinističko dopunjavanje redom 1..39 ako fali.
verify_draw_order()
Učitava sva izvlacenja, proverava da li su draw_date stringovi strogo rastući — heuristika da draw_id prati hronologiju.
get_latest_draw_date()
Poslednji zapis u listi = najnoviji; parsiranje %Y-%m-%d.
view_number_stats()
Ako nema keša u pipeline-u, vuče iz baze i pokreće analyze_number_frequency; štampa tabelu frekvencija za glavne brojeve i „powerball“ placeholder.
safe_run()
try/except oko svakog koraka — greška ne ruši ceo program, samo [ERROR] poruka.
_reset_determinism_per_run()
Pre svakog NEXT-a: random + numpy (+ opciono tf.random.set_seed), i reset globalnih tegova u config.quantum_features na deterministički normal(0, 0.1) — da ponovljeni run sa istim podacima daje isti ishod (kako komentar kaže).
run_next_combination()
clear_pipeline → ubaci historical_data → redom: istorija, frekvencija, decay, bayes, klasteri, MC, sekvenca, Markov, entropija → opciono deep learning → generate_ticket(pipeline) → vraća strukturu tiketa.
main()
initialize_database(), verify_draw_order(), petlja menija 1–5; opcija 3 validira datum (mora posle poslednjeg), 7 glavnih + bonus + placeholder PB, insert_draw, pa opet run_next_combination.
Dobre strane
Jasna orkestracija: jedan tok „svi koraci → tiket“, lako pratiti u kodu.
Otpornost na okruženje: DL se može preskočiti bez pada celog programa; isto važi za TF seed ako nema TF-a.
Determinizam: eksplicitno resetovanje seedova i kvantnih tegova — ponovljivost eksperimenata.
Bezbedniji koraci: safe_run sprečava da jedan pukli korak ubije ceo run (makar ostatak zavisi od toga da li kasniji koraci zahtevaju podatke iz puklog).
Praktični pomoćni kod: normalizacija linije na 7, provera hronologije datuma, validacija unosa za novo kolo.
Čistiji log za TensorFlow/PennyLane na Macu — smisleno za lokalni rad.
Slabe / rizične strane
Teška interpretabilnost: mnogo heterogenih signala (decay, Bayes, k-means, MC, Markov, entropija, DL, kvantni sloj u drugim modulima) — korisniku je teško da zna šta je pomerilo brojeve; rizik „crne kutije“ uz osećaj preciznosti.
safe_run: korak može „proći“ sa [OK] dok unutrašnji deo zapravo loguje grešku (u isečku na dnu fajla: Powerball clustering failed — n_samples vs n_clusters); znači tišina o delimičnom neuspehu ako step funkcija to proguta.
Zavisnost od redosleda: ako raniji korak ostavi pipeline u lošem stanju, kasniji može dati smisao bez eksplicitnog fail-fast-a.
"""





"""
Loto-739-Pipeline
Frequency Analysis. Decay Factors Calculation. Bayesian Fusion with Mechanics. Clustering and Correlation. Monte Carlo Simulations. Sequential / Temporal Features. First-Order Markov Chain. Shannon Entropy Features. Quantum Encoder Training. Quantum Features & Kernel. Deep Learning Fusion Model.
"""
