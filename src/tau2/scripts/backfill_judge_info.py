#!/usr/bin/env python3
"""
Script to backfill NL assertions and GSRT judge information in existing simulation files.

This script loads an existing simulation results file, evaluates the conversations
using NL assertions and GSRT judges, and saves the updated file with the new judge information.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.data_model.simulation import Results
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.metrics.gsrt import detect_gsrt_enhanced as detect_gsrt_v2


def analyze_missing_evaluations(results: Results) -> dict:
    """
    Analyze which simulations are missing NL assertions, GSRT evaluations, and communicate checks.
    
    Returns:
        Dictionary with missing evaluation info
    """
    missing_nl = []
    missing_gsrt = []
    missing_communicate = []
    has_nl_tasks = []
    has_communicate_tasks = []
    
    for simulation in results.simulations:
        # Find the corresponding task
        task = next((t for t in results.tasks if t.id == simulation.task_id), None)
        if task is None:
            continue
            
        # Check for missing NL assertions
        if task.evaluation_criteria and task.evaluation_criteria.nl_assertions:
            has_nl_tasks.append(simulation.id)
            if (not simulation.reward_info or 
                not simulation.reward_info.nl_assertions or 
                len(simulation.reward_info.nl_assertions) == 0):
                missing_nl.append(simulation.id)
                
        # Check for missing communicate checks
        if task.evaluation_criteria and task.evaluation_criteria.communicate_info:
            has_communicate_tasks.append(simulation.id)
            if (not simulation.reward_info or 
                not simulation.reward_info.communicate_checks or 
                len(simulation.reward_info.communicate_checks) == 0):
                missing_communicate.append(simulation.id)
                
        # Check for missing GSRT
        if (not simulation.reward_info or 
            not simulation.reward_info.info or 
            not isinstance(simulation.reward_info.info, dict) or
            "gsrt_v2" not in simulation.reward_info.info):
            missing_gsrt.append(simulation.id)
    
    return {
        "missing_nl": missing_nl,
        "missing_gsrt": missing_gsrt,
        "missing_communicate": missing_communicate,
        "has_nl_tasks": has_nl_tasks,
        "has_communicate_tasks": has_communicate_tasks,
        "total_simulations": len(results.simulations)
    }


def backfill_nl_assertions(
    results: Results,
    model: str = "gpt-4o-mini",
    llm_args: Optional[dict] = None,
    force_update: bool = False,
) -> int:
    """
    Backfill NL assertion evaluations for simulations missing them.
    
    Args:
        results: The Results object containing simulations
        model: LLM model to use for NL assertions evaluation
        llm_args: Arguments for the LLM
        force_update: If True, update even if NL assertions already exist
        
    Returns:
        Number of simulations updated
    """
    if llm_args is None:
        llm_args = {"temperature": 0.0}
        
    updated_count = 0
    
    for simulation in results.simulations:
        # Find the corresponding task
        task = next((t for t in results.tasks if t.id == simulation.task_id), None)
        if task is None:
            logger.warning(f"Task {simulation.task_id} not found for simulation {simulation.id}")
            continue
            
        # Check if task has NL assertions to evaluate
        if not task.evaluation_criteria or not task.evaluation_criteria.nl_assertions:
            continue
            
        # Check if NL assertions are already present and valid
        has_nl_assertions = (simulation.reward_info and 
                           simulation.reward_info.nl_assertions and 
                           len(simulation.reward_info.nl_assertions) > 0)
        
        if has_nl_assertions and not force_update:
            logger.debug(f"NL assertions already present for simulation {simulation.id}, skipping")
            continue
            
        try:
            logger.info(f"Evaluating NL assertions for simulation {simulation.id} (task: {simulation.task_id})")
            nl_assertions_checks = NLAssertionsEvaluator.evaluate_nl_assertions(
                simulation.messages, 
                task.evaluation_criteria.nl_assertions,
                model,
                llm_args
            )
            
            # Update the simulation with NL assertion results
            if simulation.reward_info is None:
                from tau2.data_model.simulation import RewardInfo
                simulation.reward_info = RewardInfo(reward=0.0)
                
            simulation.reward_info.nl_assertions = nl_assertions_checks
            updated_count += 1
            logger.info(f"✅ Updated NL assertions for simulation {simulation.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate NL assertions for simulation {simulation.id}: {e}")
            
    return updated_count


def backfill_communicate_checks(
    results: Results,
    force_update: bool = False,
) -> int:
    """
    Backfill communicate checks for simulations missing them.
    
    Args:
        results: The Results object containing simulations
        force_update: If True, update even if communicate checks already exist
        
    Returns:
        Number of simulations updated
    """
    updated_count = 0
    
    for simulation in results.simulations:
        # Find the corresponding task
        task = next((t for t in results.tasks if t.id == simulation.task_id), None)
        if task is None:
            logger.warning(f"Task {simulation.task_id} not found for simulation {simulation.id}")
            continue
            
        # Check if task has communicate_info to evaluate
        if not task.evaluation_criteria or not task.evaluation_criteria.communicate_info:
            continue
            
        # Check if communicate checks are already present and valid
        has_communicate_checks = (simulation.reward_info and 
                                simulation.reward_info.communicate_checks and 
                                len(simulation.reward_info.communicate_checks) > 0)
        
        if has_communicate_checks and not force_update:
            logger.debug(f"Communicate checks already present for simulation {simulation.id}, skipping")
            continue
            
        try:
            logger.info(f"Evaluating communicate checks for simulation {simulation.id} (task: {simulation.task_id})")
            communicate_checks = CommunicateEvaluator.evaluate_communicate_info(
                simulation.messages, 
                task.evaluation_criteria.communicate_info
            )
            
            # Update the simulation with communicate check results
            if simulation.reward_info is None:
                from tau2.data_model.simulation import RewardInfo
                simulation.reward_info = RewardInfo(reward=0.0)
                
            simulation.reward_info.communicate_checks = communicate_checks
            updated_count += 1
            logger.info(f"✅ Updated communicate checks for simulation {simulation.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate communicate checks for simulation {simulation.id}: {e}")
            
    return updated_count


def backfill_gsrt_judge(
    results: Results,
    model: str = "gpt-4o-mini", 
    llm_args: Optional[dict] = None,
    force_update: bool = False,
) -> int:
    """
    Backfill GSRT judge evaluations for simulations missing them.
    
    Args:
        results: The Results object containing simulations
        model: LLM model to use for GSRT evaluation
        llm_args: Arguments for the LLM
        force_update: If True, update even if GSRT info already exists
        
    Returns:
        Number of simulations updated
    """
    if llm_args is None:
        llm_args = {"temperature": 0.0}
        
    updated_count = 0
    
    for simulation in results.simulations:
        # Find the corresponding task
        task = next((t for t in results.tasks if t.id == simulation.task_id), None)
        if task is None:
            logger.warning(f"Task {simulation.task_id} not found for simulation {simulation.id}")
            continue
            
        # Check if GSRT judge info is already present
        has_gsrt = (simulation.reward_info and 
                   simulation.reward_info.info and 
                   isinstance(simulation.reward_info.info, dict) and
                   "gsrt_v2" in simulation.reward_info.info)
        
        if has_gsrt and not force_update:
            logger.debug(f"GSRT judge info already present for simulation {simulation.id}, skipping")
            continue
            
        try:
            logger.info(f"Evaluating GSRT for simulation {simulation.id} (task: {simulation.task_id})")
            gsrt_result = detect_gsrt_v2(task, simulation, model=model, llm_args=llm_args)
            
            # Update the simulation with GSRT results
            if simulation.reward_info is None:
                from tau2.data_model.simulation import RewardInfo
                simulation.reward_info = RewardInfo(reward=0.0)
                
            if simulation.reward_info.info is None:
                simulation.reward_info.info = {}
                
            if isinstance(simulation.reward_info.info, dict):
                simulation.reward_info.info["gsrt_v2"] = gsrt_result
                updated_count += 1
                logger.info(f"✅ Updated GSRT judge info for simulation {simulation.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate GSRT for simulation {simulation.id}: {e}")
            
    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Intelligently backfill missing NL assertions, communicate checks, and GSRT judge information in simulation files"
    )
    parser.add_argument(
        "simulation_file",
        type=Path,
        help="Path to the simulation results JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: creates <original>_complete.json)"
    )
    parser.add_argument(
        "--nl-model",
        default="gpt-4o-mini",
        help="LLM model to use for NL assertions evaluation (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--gsrt-model", 
        default="gpt-4o-mini",
        help="LLM model to use for GSRT evaluation (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--nl-only",
        action="store_true",
        help="Only backfill NL assertions"
    )
    parser.add_argument(
        "--gsrt-only",
        action="store_true",
        help="Only backfill GSRT judge info"
    )
    parser.add_argument(
        "--communicate-only",
        action="store_true",
        help="Only backfill communicate checks"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update even if judge info already exists"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM evaluation (default: 0.0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze what's missing without making changes"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.simulation_file.exists():
        logger.error(f"Simulation file {args.simulation_file} does not exist")
        return
        
    # Set output file - create a complete version by default
    if args.output:
        output_file = args.output
    else:
        base_name = args.simulation_file.stem
        output_file = args.simulation_file.parent / f"{base_name}_complete.json"
    
    # Load simulation results
    logger.info(f"📂 Loading simulation results from {args.simulation_file}")
    results = Results.load(args.simulation_file)
    logger.info(f"📊 Loaded {len(results.simulations)} simulations")
    
    # Analyze what's missing
    logger.info(f"🔍 Analyzing missing evaluations...")
    analysis = analyze_missing_evaluations(results)
    
    print(f"\n{'='*60}")
    print(f"📋 ANALYSIS REPORT")
    print(f"{'='*60}")
    print(f"Total simulations: {analysis['total_simulations']}")
    print(f"Tasks with NL assertions: {len(analysis['has_nl_tasks'])}")
    print(f"Tasks with communicate info: {len(analysis['has_communicate_tasks'])}")
    print(f"Missing NL assertions: {len(analysis['missing_nl'])}")
    print(f"Missing communicate checks: {len(analysis['missing_communicate'])}")
    print(f"Missing GSRT evaluations: {len(analysis['missing_gsrt'])}")
    
    if analysis['missing_nl']:
        print(f"\n🔍 Simulations missing NL assertions:")
        for sim_id in analysis['missing_nl'][:10]:  # Show first 10
            print(f"  - {sim_id}")
        if len(analysis['missing_nl']) > 10:
            print(f"  ... and {len(analysis['missing_nl']) - 10} more")
    
    if analysis['missing_communicate']:
        print(f"\n🔍 Simulations missing communicate checks:")
        for sim_id in analysis['missing_communicate'][:10]:  # Show first 10
            print(f"  - {sim_id}")
        if len(analysis['missing_communicate']) > 10:
            print(f"  ... and {len(analysis['missing_communicate']) - 10} more")
    
    if analysis['missing_gsrt']:
        print(f"\n🔍 Simulations missing GSRT evaluations:")
        for sim_id in analysis['missing_gsrt'][:10]:  # Show first 10
            print(f"  - {sim_id}")
        if len(analysis['missing_gsrt']) > 10:
            print(f"  ... and {len(analysis['missing_gsrt']) - 10} more")
    
    if args.dry_run:
        print(f"\n✅ Dry run complete - no changes made")
        return
    
    if not analysis['missing_nl'] and not analysis['missing_gsrt'] and not analysis['missing_communicate']:
        print(f"\n✅ All judge information is already present!")
        return
    
    print(f"\n🚀 Starting backfill process...")
    
    # Prepare LLM args
    llm_args = {"temperature": args.temperature}
    
    # Backfill judge information
    nl_updated = 0
    gsrt_updated = 0
    communicate_updated = 0
    
    # Check which evaluations to run based on flags
    run_nl = not (args.gsrt_only or args.communicate_only) and analysis['missing_nl']
    run_communicate = not (args.nl_only or args.gsrt_only) and analysis['missing_communicate']
    run_gsrt = not (args.nl_only or args.communicate_only) and analysis['missing_gsrt']
    
    if run_nl:
        print(f"\n📝 Backfilling {len(analysis['missing_nl'])} missing NL assertions...")
        nl_updated = backfill_nl_assertions(results, args.nl_model, llm_args, args.force)
        print(f"✅ Updated NL assertions for {nl_updated} simulations")
        
    if run_communicate:
        print(f"\n💬 Backfilling {len(analysis['missing_communicate'])} missing communicate checks...")
        communicate_updated = backfill_communicate_checks(results, args.force)
        print(f"✅ Updated communicate checks for {communicate_updated} simulations")
        
    if run_gsrt:
        print(f"\n🎯 Backfilling {len(analysis['missing_gsrt'])} missing GSRT evaluations...")
        gsrt_updated = backfill_gsrt_judge(results, args.gsrt_model, llm_args, args.force)
        print(f"✅ Updated GSRT judge info for {gsrt_updated} simulations")
    
    # Save updated results
    if nl_updated > 0 or gsrt_updated > 0 or communicate_updated > 0:
        print(f"\n💾 Saving complete results to {output_file}")
        results.save(output_file)
        print(f"🎉 Backfill completed successfully!")
        print(f"📁 Complete simulation file: {output_file}")
        
        # Final analysis
        final_analysis = analyze_missing_evaluations(results)
        print(f"\n📊 FINAL STATUS:")
        print(f"  Missing NL assertions: {len(final_analysis['missing_nl'])}")
        print(f"  Missing communicate checks: {len(final_analysis['missing_communicate'])}")
        print(f"  Missing GSRT evaluations: {len(final_analysis['missing_gsrt'])}")
    else:
        print(f"\n⚠️  No simulations were updated")
        if not args.force:
            print(f"💡 Use --force to update existing judge information")


if __name__ == "__main__":
    main()
