from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .api import AstarIslandClient
from .config import load_env_file
from .dataset import export_training_dataset, summarize_round
from .prediction import SeedPredictor
from .storage import DataStore
from .strategy import round_robin_schedule
from .training import (
    DEFAULT_LIVE_MODEL_WEIGHT,
    DEFAULT_NO_OBSERVATION_MODEL_WEIGHT,
    DEFAULT_OBSERVATION_HALF_LIFE,
    load_ridge_probability_model,
    predict_round_with_model,
    train_and_save_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Astar Island competition toolkit")
    parser.add_argument("--token", default=None, help="JWT token. Defaults to ASTAR_ISLAND_TOKEN.")
    parser.add_argument("--base-url", default="https://api.ainm.no/astar-island")
    parser.add_argument("--data-root", default="data", help="Directory for archived round data.")
    parser.add_argument("--timeout", type=float, default=30.0)

    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("round-info", help="Fetch and archive round details.")
    info_parser.add_argument("--round-id", default=None)
    info_parser.set_defaults(handler=handle_round_info)

    budget_parser = subparsers.add_parser("budget", help="Fetch current team budget for the active round.")
    budget_parser.set_defaults(handler=handle_budget)

    my_rounds_parser = subparsers.add_parser("my-rounds", help="Fetch team-specific round status and scores.")
    my_rounds_parser.set_defaults(handler=handle_my_rounds)

    my_predictions_parser = subparsers.add_parser(
        "my-predictions",
        help="Fetch team submissions for a round.",
    )
    my_predictions_parser.add_argument("--round-id", required=True)
    my_predictions_parser.set_defaults(handler=handle_my_predictions)

    simulate_parser = subparsers.add_parser("simulate", help="Run and archive one observation query.")
    simulate_parser.add_argument("--round-id", default=None)
    simulate_parser.add_argument("--seed-index", type=int, required=True)
    simulate_parser.add_argument("--x", type=int, required=True)
    simulate_parser.add_argument("--y", type=int, required=True)
    simulate_parser.add_argument("--w", type=int, default=15)
    simulate_parser.add_argument("--h", type=int, default=15)
    simulate_parser.set_defaults(handler=handle_simulate)

    run_parser = subparsers.add_parser(
        "run-baseline",
        help="Build a baseline prediction, optionally observe, and submit for each seed.",
    )
    run_parser.add_argument("--round-id", default=None)
    run_parser.add_argument("--max-observations-per-seed", type=int, default=1)
    run_parser.add_argument("--viewport-size", type=int, default=15)
    run_parser.add_argument(
        "--simulate-delay-seconds",
        type=float,
        default=0.25,
        help="Delay between simulate calls to respect API rate limits.",
    )
    run_parser.add_argument(
        "--submit-delay-seconds",
        type=float,
        default=0.6,
        help="Delay between submit calls to respect API rate limits.",
    )
    run_parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Save predictions locally without calling /submit.",
    )
    run_parser.set_defaults(handler=handle_run_baseline)

    sim_parser = subparsers.add_parser(
        "run-simulate",
        help="Build prediction using Monte Carlo simulation and submit.",
    )
    sim_parser.add_argument("--round-id", default=None)
    sim_parser.add_argument("--max-observations-per-seed", type=int, default=10)
    sim_parser.add_argument("--viewport-size", type=int, default=15)
    sim_parser.add_argument("--simulate-runs", type=int, default=30)
    sim_parser.add_argument("--simulate-delay-seconds", type=float, default=0.25)
    sim_parser.add_argument("--submit-delay-seconds", type=float, default=0.6)
    sim_parser.add_argument("--no-submit", action="store_true")
    sim_parser.set_defaults(handler=handle_run_simulate)

    analysis_parser = subparsers.add_parser(
        "fetch-analysis",
        help="Archive post-round analysis data for one or all seeds.",
    )
    analysis_parser.add_argument("--round-id", required=True)
    analysis_parser.add_argument("--seed-index", type=int, default=None)
    analysis_parser.set_defaults(handler=handle_fetch_analysis)

    sync_parser = subparsers.add_parser(
        "sync-api",
        help="Archive all accessible non-destructive API data for offline analysis.",
    )
    sync_parser.set_defaults(handler=handle_sync_api)

    analyze_parser = subparsers.add_parser(
        "analyze-round",
        help="Summarize archived predictions versus ground truth for a completed round.",
    )
    analyze_parser.add_argument("--round-id", required=True)
    analyze_parser.set_defaults(handler=handle_analyze_round)

    export_parser = subparsers.add_parser(
        "export-dataset",
        help="Export completed rounds into stacked NPZ arrays for offline training.",
    )
    export_parser.add_argument(
        "--round-id",
        action="append",
        default=None,
        help="Round ID to include. Repeat to include multiple rounds. Defaults to all completed rounds in data root.",
    )
    export_parser.add_argument(
        "--output",
        default="data/training_dataset.npz",
        help="Path to the exported dataset .npz file.",
    )
    export_parser.set_defaults(handler=handle_export_dataset)

    train_parser = subparsers.add_parser(
        "train-model",
        help="Train a lightweight offline baseline model from an exported dataset.",
    )
    train_parser.add_argument(
        "--dataset",
        default="data/training_dataset.npz",
        help="Path to the exported training dataset .npz file.",
    )
    train_parser.add_argument(
        "--output",
        default="data/models/ridge_probability_model.npz",
        help="Path to the saved model .npz artifact.",
    )
    train_parser.add_argument("--reg-strength", type=float, default=1.0)
    train_parser.add_argument("--entropy-weight-scale", type=float, default=3.0)
    train_parser.set_defaults(handler=handle_train_model)

    model_parser = subparsers.add_parser(
        "run-model",
        help="Build a model-based prediction, optionally observe, and submit for each seed.",
    )
    model_parser.add_argument(
        "--model",
        default="data/models/ridge_probability_model.npz",
        help="Path to the saved model .npz artifact.",
    )
    model_parser.add_argument("--round-id", default=None)
    model_parser.add_argument("--max-observations-per-seed", type=int, default=0)
    model_parser.add_argument("--viewport-size", type=int, default=15)
    model_parser.add_argument(
        "--model-weight",
        type=float,
        default=DEFAULT_LIVE_MODEL_WEIGHT,
        help="Maximum model correction weight on top of the empirical baseline.",
    )
    model_parser.add_argument(
        "--observation-half-life",
        type=float,
        default=DEFAULT_OBSERVATION_HALF_LIFE,
        help="Observation count where model influence is halved for a cell.",
    )
    model_parser.add_argument(
        "--no-observation-weight",
        type=float,
        default=DEFAULT_NO_OBSERVATION_MODEL_WEIGHT,
        help="Weight for the zero-query model branch in the final live ensemble.",
    )
    model_parser.add_argument(
        "--simulate-delay-seconds",
        type=float,
        default=0.25,
        help="Delay between simulate calls to respect API rate limits.",
    )
    model_parser.add_argument(
        "--submit-delay-seconds",
        type=float,
        default=0.6,
        help="Delay between submit calls to respect API rate limits.",
    )
    model_parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Save predictions locally without calling /submit.",
    )
    model_parser.set_defaults(handler=handle_run_model)

    return parser


def build_client(args: argparse.Namespace) -> AstarIslandClient:
    return AstarIslandClient.from_env(
        token=args.token,
        base_url=args.base_url,
        timeout=args.timeout,
    )


def build_store(args: argparse.Namespace) -> DataStore:
    return DataStore(Path(args.data_root))


def resolve_round(
    client: AstarIslandClient,
    round_id: str | None,
    *,
    store: DataStore | None = None,
    prefer_cache: bool = False,
) -> dict[str, Any]:
    if round_id:
        if prefer_cache and store is not None:
            try:
                return store.load_round_details(round_id)
            except FileNotFoundError:
                pass
        return client.get_round_details(round_id)

    active_round = client.get_active_round()
    if not active_round:
        raise SystemExit("No active round found.")
    return client.get_round_details(str(active_round["id"]))


def maybe_get_budget_for_round(
    client: AstarIslandClient,
    round_id: str,
    *,
    store: DataStore | None = None,
) -> dict[str, Any] | None:
    try:
        budget = client.get_budget()
    except Exception:
        budget = None
    if budget is not None and str(budget.get("round_id", "")) == round_id:
        return budget
    if store is not None:
        cached_budget = store.load_latest_budget(round_id)
        if cached_budget is not None:
            return cached_budget
    return None


def handle_round_info(args: argparse.Namespace) -> None:
    client = build_client(args)
    store = build_store(args)
    round_details = resolve_round(client, args.round_id)
    store.save_round_details(round_details)
    print(json.dumps(_round_summary(round_details), indent=2))


def handle_budget(args: argparse.Namespace) -> None:
    client = build_client(args)
    budget = client.get_budget()
    print(json.dumps(budget, indent=2))


def handle_my_rounds(args: argparse.Namespace) -> None:
    client = build_client(args)
    rounds = client.get_my_rounds()
    print(json.dumps(rounds, indent=2))


def handle_my_predictions(args: argparse.Namespace) -> None:
    client = build_client(args)
    predictions = client.get_my_predictions(args.round_id)
    print(json.dumps(predictions, indent=2))


def handle_simulate(args: argparse.Namespace) -> None:
    client = build_client(args)
    store = build_store(args)
    round_details = resolve_round(client, args.round_id)
    round_id = str(round_details["id"])
    store.save_round_details(round_details)

    request_payload = {
        "round_id": round_id,
        "seed_index": args.seed_index,
        "viewport_x": args.x,
        "viewport_y": args.y,
        "viewport_w": args.w,
        "viewport_h": args.h,
    }
    response_payload = client.simulate(
        round_id=round_id,
        seed_index=args.seed_index,
        viewport_x=args.x,
        viewport_y=args.y,
        viewport_w=args.w,
        viewport_h=args.h,
    )
    store.save_observation(round_id, request_payload, response_payload)
    store.save_budget(
        round_id,
        {
            "queries_used": response_payload.get("queries_used"),
            "queries_max": response_payload.get("queries_max"),
        },
    )
    print(json.dumps(_simulate_summary(response_payload), indent=2))


def handle_run_baseline(args: argparse.Namespace) -> None:
    from .dynamics import extract_dynamics

    client = build_client(args)
    store = build_store(args)
    round_details = resolve_round(client, args.round_id, store=store, prefer_cache=True)
    round_id = str(round_details["id"])
    store.save_round_details(round_details)

    budget = maybe_get_budget_for_round(client, round_id, store=store)
    if budget is not None:
        store.save_budget(round_id, budget)

    existing_observations = store.load_observations(round_id)
    existing_observations_by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in existing_observations:
        request_payload = record.get("request", {})
        seed_index = int(request_payload["seed_index"])
        existing_observations_by_seed.setdefault(seed_index, []).append(record)

    # Phase 1: Run observations
    if args.max_observations_per_seed > 0:
        if budget is None:
            raise SystemExit("Budget unavailable for this round; cannot run live observations.")
        schedule = round_robin_schedule(
            round_details,
            max_observations_per_seed=args.max_observations_per_seed,
            existing_observations_by_seed=existing_observations_by_seed,
            viewport_size=args.viewport_size,
        )
        remaining_budget = int(budget["queries_max"]) - int(budget["queries_used"])
        schedule = schedule[: max(0, remaining_budget)]

        for item in schedule:
            request_payload = {
                "round_id": round_id,
                "seed_index": item["seed_index"],
                "viewport_x": item["x"],
                "viewport_y": item["y"],
                "viewport_w": item["w"],
                "viewport_h": item["h"],
            }
            response_payload = client.simulate(
                round_id=round_id,
                seed_index=item["seed_index"],
                viewport_x=item["x"],
                viewport_y=item["y"],
                viewport_w=item["w"],
                viewport_h=item["h"],
            )
            record = {"request": request_payload, "response": response_payload}
            existing_observations.append(record)
            existing_observations_by_seed.setdefault(item["seed_index"], []).append(record)
            store.save_observation(round_id, request_payload, response_payload)
            store.save_budget(
                round_id,
                {
                    "queries_used": response_payload.get("queries_used"),
                    "queries_max": response_payload.get("queries_max"),
                },
            )
            time.sleep(max(0.0, args.simulate_delay_seconds))

    # Phase 2: Extract dynamics from ALL observations and rebuild predictors
    dynamics = extract_dynamics(existing_observations)
    predictors = [
        SeedPredictor.from_initial_state(initial_state, dynamics=dynamics)
        for initial_state in round_details["initial_states"]
    ]
    for record in existing_observations:
        request_payload = record.get("request", {})
        response_payload = record.get("response", {})
        seed_index = int(request_payload["seed_index"])
        predictors[seed_index].observe(response_payload)

    submitted = 0
    saved_predictions = []
    for seed_index, predictor in enumerate(predictors):
        prediction = predictor.prediction()
        metadata: dict[str, Any] = {
            "submitted": not args.no_submit,
            "dynamics": {
                "settlement_alive_rate": dynamics.settlement_alive_rate,
                "is_collapse": dynamics.is_collapse,
                "is_thriving": dynamics.is_thriving,
                "observed_queries": dynamics.observed_queries,
            },
        }
        if args.no_submit:
            submission_result = None
        else:
            submission_result = client.submit(round_id, seed_index, prediction.tolist())
            submitted += 1
            metadata["submission_result"] = submission_result
            if seed_index + 1 < len(predictors):
                time.sleep(max(0.0, args.submit_delay_seconds))

        path = store.save_prediction(round_id, seed_index, prediction, metadata=metadata)
        saved_predictions.append(str(path))

    print(
        json.dumps(
            {
                "round_id": round_id,
                "submitted": submitted,
                "dynamics": {
                    "settlement_alive_rate": dynamics.settlement_alive_rate,
                    "is_collapse": dynamics.is_collapse,
                    "is_thriving": dynamics.is_thriving,
                },
                "saved_predictions": saved_predictions,
            },
            indent=2,
        )
    )


def handle_run_simulate(args: argparse.Namespace) -> None:
    from .dynamics import extract_dynamics

    client = build_client(args)
    store = build_store(args)
    round_details = resolve_round(client, args.round_id, store=store, prefer_cache=True)
    round_id = str(round_details["id"])
    store.save_round_details(round_details)

    budget = maybe_get_budget_for_round(client, round_id, store=store)
    if budget is not None:
        store.save_budget(round_id, budget)

    existing_observations = store.load_observations(round_id)
    existing_observations_by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in existing_observations:
        request_payload = record.get("request", {})
        seed_index = int(request_payload["seed_index"])
        existing_observations_by_seed.setdefault(seed_index, []).append(record)

    # Phase 1: Run observations
    if args.max_observations_per_seed > 0:
        if budget is None:
            raise SystemExit("Budget unavailable for this round; cannot run live observations.")
        schedule = round_robin_schedule(
            round_details,
            max_observations_per_seed=args.max_observations_per_seed,
            existing_observations_by_seed=existing_observations_by_seed,
            viewport_size=args.viewport_size,
        )
        remaining_budget = int(budget["queries_max"]) - int(budget["queries_used"])
        schedule = schedule[: max(0, remaining_budget)]

        for item in schedule:
            request_payload = {
                "round_id": round_id,
                "seed_index": item["seed_index"],
                "viewport_x": item["x"],
                "viewport_y": item["y"],
                "viewport_w": item["w"],
                "viewport_h": item["h"],
            }
            response_payload = client.simulate(
                round_id=round_id,
                seed_index=item["seed_index"],
                viewport_x=item["x"],
                viewport_y=item["y"],
                viewport_w=item["w"],
                viewport_h=item["h"],
            )
            record = {"request": request_payload, "response": response_payload}
            existing_observations.append(record)
            existing_observations_by_seed.setdefault(item["seed_index"], []).append(record)
            store.save_observation(round_id, request_payload, response_payload)
            store.save_budget(
                round_id,
                {
                    "queries_used": response_payload.get("queries_used"),
                    "queries_max": response_payload.get("queries_max"),
                },
            )
            time.sleep(max(0.0, args.simulate_delay_seconds))

    # Phase 2: Extract dynamics and build simulation-based predictions
    dynamics = extract_dynamics(existing_observations)
    predictors = [
        SeedPredictor.from_initial_state(
            initial_state, dynamics=dynamics,
            simulate=True, simulate_runs=args.simulate_runs,
        )
        for initial_state in round_details["initial_states"]
    ]
    for record in existing_observations:
        request_payload = record.get("request", {})
        response_payload = record.get("response", {})
        seed_index = int(request_payload["seed_index"])
        predictors[seed_index].observe(response_payload)

    submitted = 0
    saved_predictions = []
    for seed_index, predictor in enumerate(predictors):
        prediction = predictor.prediction()
        metadata: dict[str, Any] = {
            "submitted": not args.no_submit,
            "mode": "simulate",
            "simulate_runs": args.simulate_runs,
            "dynamics": {
                "settlement_alive_rate": dynamics.settlement_alive_rate,
                "is_collapse": dynamics.is_collapse,
                "is_thriving": dynamics.is_thriving,
                "observed_queries": dynamics.observed_queries,
            },
        }
        if args.no_submit:
            submission_result = None
        else:
            submission_result = client.submit(round_id, seed_index, prediction.tolist())
            submitted += 1
            metadata["submission_result"] = submission_result
            if seed_index + 1 < len(predictors):
                time.sleep(max(0.0, args.submit_delay_seconds))

        path = store.save_prediction(round_id, seed_index, prediction, metadata=metadata)
        saved_predictions.append(str(path))

    print(
        json.dumps(
            {
                "round_id": round_id,
                "submitted": submitted,
                "mode": "simulate",
                "simulate_runs": args.simulate_runs,
                "dynamics": {
                    "settlement_alive_rate": dynamics.settlement_alive_rate,
                    "is_collapse": dynamics.is_collapse,
                    "is_thriving": dynamics.is_thriving,
                },
                "saved_predictions": saved_predictions,
            },
            indent=2,
        )
    )


def handle_fetch_analysis(args: argparse.Namespace) -> None:
    client = build_client(args)
    store = build_store(args)
    round_details = client.get_round_details(args.round_id)
    store.save_round_details(round_details)

    seeds_count = int(round_details["seeds_count"])
    seed_indices = [args.seed_index] if args.seed_index is not None else list(range(seeds_count))
    saved = []
    for seed_index in seed_indices:
        analysis = client.get_analysis(args.round_id, seed_index)
        saved.append(str(store.save_analysis(args.round_id, seed_index, analysis)))

    print(json.dumps({"round_id": args.round_id, "saved": saved}, indent=2))


def handle_sync_api(args: argparse.Namespace) -> None:
    client = build_client(args)
    store = build_store(args)

    rounds = client.list_rounds()
    my_rounds = client.get_my_rounds()
    leaderboard = client.get_leaderboard()

    saved: dict[str, Any] = {
        "snapshots": [],
        "round_details": 0,
        "team_rounds": 0,
        "my_predictions": 0,
        "analysis": 0,
        "budget": 0,
    }

    saved["snapshots"].append(str(store.save_snapshot("rounds.json", rounds)))
    saved["snapshots"].append(str(store.save_snapshot("my_rounds.json", my_rounds)))
    saved["snapshots"].append(str(store.save_snapshot("leaderboard.json", leaderboard)))

    round_ids = sorted({str(item["id"]) for item in rounds} | {str(item["id"]) for item in my_rounds})
    rounds_by_id = {str(item["id"]): item for item in rounds}
    my_rounds_by_id = {str(item["id"]): item for item in my_rounds}

    for round_id in round_ids:
        round_details = client.get_round_details(round_id)
        store.save_round_details(round_details)
        saved["round_details"] += 1

        team_round = my_rounds_by_id.get(round_id)
        if team_round is not None:
            store.save_round_json(round_id, "team_round.json", team_round)
            saved["team_rounds"] += 1

        status = str(rounds_by_id.get(round_id, {}).get("status", round_details.get("status", "")))
        if status == "active":
            try:
                budget = client.get_budget()
            except Exception:
                budget = None
            if budget is not None and str(budget.get("round_id")) == round_id:
                store.save_budget(round_id, budget)
                store.save_round_json(round_id, "active_budget.json", budget)
                saved["budget"] += 1

        seeds_submitted = int(team_round.get("seeds_submitted", 0)) if team_round else 0
        if seeds_submitted > 0:
            predictions = client.get_my_predictions(round_id)
            store.save_round_json(round_id, "my_predictions.json", predictions)
            saved["my_predictions"] += 1

            if status in {"scoring", "completed"}:
                for item in predictions:
                    seed_index = int(item["seed_index"])
                    analysis = client.get_analysis(round_id, seed_index)
                    store.save_analysis(round_id, seed_index, analysis)
                    saved["analysis"] += 1

    print(json.dumps(saved, indent=2))


def handle_analyze_round(args: argparse.Namespace) -> None:
    store = build_store(args)
    print(json.dumps(summarize_round(store, args.round_id), indent=2))


def handle_export_dataset(args: argparse.Namespace) -> None:
    store = build_store(args)
    result = export_training_dataset(
        store,
        args.output,
        round_ids=args.round_id,
    )
    print(json.dumps(result, indent=2))


def handle_train_model(args: argparse.Namespace) -> None:
    result = train_and_save_model(
        args.dataset,
        args.output,
        reg_strength=args.reg_strength,
        entropy_weight_scale=args.entropy_weight_scale,
    )
    print(json.dumps(result, indent=2))


def handle_run_model(args: argparse.Namespace) -> None:
    client = build_client(args)
    store = build_store(args)
    round_details = resolve_round(client, args.round_id, store=store, prefer_cache=True)
    round_id = str(round_details["id"])
    store.save_round_details(round_details)

    budget = maybe_get_budget_for_round(client, round_id, store=store)
    if budget is not None:
        store.save_budget(round_id, budget)

    existing_observations = store.load_observations(round_id)
    existing_observations_by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in existing_observations:
        request_payload = record.get("request", {})
        seed_index = int(request_payload["seed_index"])
        existing_observations_by_seed.setdefault(seed_index, []).append(record)

    if args.max_observations_per_seed > 0:
        if budget is None:
            raise SystemExit("Budget unavailable for this round; cannot run live observations.")
        schedule = round_robin_schedule(
            round_details,
            max_observations_per_seed=args.max_observations_per_seed,
            existing_observations_by_seed=existing_observations_by_seed,
            viewport_size=args.viewport_size,
        )
        remaining_budget = int(budget["queries_max"]) - int(budget["queries_used"])
        schedule = schedule[: max(0, remaining_budget)]

        for item in schedule:
            request_payload = {
                "round_id": round_id,
                "seed_index": item["seed_index"],
                "viewport_x": item["x"],
                "viewport_y": item["y"],
                "viewport_w": item["w"],
                "viewport_h": item["h"],
            }
            response_payload = client.simulate(
                round_id=round_id,
                seed_index=item["seed_index"],
                viewport_x=item["x"],
                viewport_y=item["y"],
                viewport_w=item["w"],
                viewport_h=item["h"],
            )
            record = {
                "request": request_payload,
                "response": response_payload,
            }
            existing_observations.append(record)
            existing_observations_by_seed.setdefault(item["seed_index"], []).append(record)
            store.save_observation(round_id, request_payload, response_payload)
            store.save_budget(
                round_id,
                {
                    "queries_used": response_payload.get("queries_used"),
                    "queries_max": response_payload.get("queries_max"),
                },
            )
            time.sleep(max(0.0, args.simulate_delay_seconds))

    model = load_ridge_probability_model(args.model)
    predictions = predict_round_with_model(
        model,
        round_details,
        existing_observations,
        max_model_weight=args.model_weight,
        observation_half_life=args.observation_half_life,
        no_observation_model_weight=args.no_observation_weight,
    )

    submitted = 0
    saved_predictions = []
    for seed_index, prediction in enumerate(predictions):
        metadata: dict[str, Any] = {
            "submitted": not args.no_submit,
            "model_path": args.model,
            "model_weight": args.model_weight,
            "observation_half_life": args.observation_half_life,
            "no_observation_weight": args.no_observation_weight,
            "observation_count": len(existing_observations_by_seed.get(seed_index, [])),
        }
        if args.no_submit:
            submission_result = None
        else:
            submission_result = client.submit(round_id, seed_index, prediction.tolist())
            submitted += 1
            metadata["submission_result"] = submission_result
            if seed_index + 1 < len(predictions):
                time.sleep(max(0.0, args.submit_delay_seconds))

        path = store.save_prediction(round_id, seed_index, prediction, metadata=metadata)
        saved_predictions.append(str(path))

    print(
        json.dumps(
            {
                "round_id": round_id,
                "model": args.model,
                "submitted": submitted,
                "saved_predictions": saved_predictions,
            },
            indent=2,
        )
    )


def _round_summary(round_details: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": round_details["id"],
        "round_number": round_details["round_number"],
        "status": round_details["status"],
        "map_width": round_details["map_width"],
        "map_height": round_details["map_height"],
        "seeds_count": round_details["seeds_count"],
    }


def _simulate_summary(response_payload: dict[str, Any]) -> dict[str, Any]:
    viewport = response_payload["viewport"]
    return {
        "viewport": viewport,
        "queries_used": response_payload.get("queries_used"),
        "queries_max": response_payload.get("queries_max"),
        "settlements_visible": len(response_payload.get("settlements", [])),
        "grid_rows": len(response_payload.get("grid", [])),
        "grid_cols": len(response_payload.get("grid", [[]])[0]) if response_payload.get("grid") else 0,
    }


def main() -> None:
    load_env_file()
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
