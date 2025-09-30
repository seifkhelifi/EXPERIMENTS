import os

import time
import json
from typing import Dict, Any
from statistics import mean, median, stdev
from datetime import datetime

from main import IoTSOCAgent

from dotenv import load_dotenv

load_dotenv()


def test_models_latency(
    api_key: str, test_flow: str = None, output_file: str = "latency_results.json"
) -> Dict[str, Any]:

    if test_flow is None:
        test_flow = "Network flow from IoT device 192.168.1.101 to server 10.0.0.2 on port 80 using TCP, duration 0.1 seconds, 1 forward packet, 0 backward packets, 64 bytes total. SYN flag set, no FIN flag."

    models = [
        "deepseek-r1-distill-llama-70b",
        "gemma2-9b-it",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-guard-4-12b",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",
        "moonshotai/kimi-k2-instruct",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b",
    ]

    results = {
        "test_metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_flow": test_flow,
            "total_models_tested": 0,
            "successful_tests": 0,
            "failed_tests": 0,
        },
        "model_results": {},
        "summary": {},
    }

    print(f"🚀 Starting latency testing for {len(models)} models...")
    print(f"📊 Test flow: {test_flow[:100]}...")

    for model_name in models:
        print(f"\n🔧 Testing model: {model_name}")
        results["test_metadata"]["total_models_tested"] += 1

        try:
            # Create agent instance with current model
            agent = IoTSOCAgent(api_key, model=model_name)

            # Track latencies for this model
            model_latencies = []
            step_details = []
            flow_start_time = time.time()

            # Modified analyze_flow to capture latencies
            context = ""
            iteration = 0

            while iteration < agent.max_iterations:
                iteration += 1

                # Measure individual LLM call latency
                call_start_time = time.time()

                try:
                    security_step = agent.get_structured_response(test_flow, context)
                    call_end_time = time.time()
                    call_latency = call_end_time - call_start_time

                    model_latencies.append(call_latency)
                    step_details.append(
                        {
                            "step": iteration,
                            "action_type": security_step.action.action_type,
                            "latency_seconds": round(call_latency, 4),
                            "thought_length": len(security_step.thought),
                            "reasoning_length": len(security_step.action.reasoning),
                        }
                    )

                    print(
                        f"  Step {iteration}: {security_step.action.action_type} - {call_latency:.3f}s"
                    )

                    # Execute the action
                    result = agent.execute_action(security_step.action)

                    # If this is a respond action, we're done
                    if isinstance(security_step.action, RespondAction):
                        break

                    context += f"\nStep {iteration} - Action: {security_step.action.action_type} | Result: {result}"

                except Exception as step_error:
                    print(f"  ⚠️  Step {iteration} failed: {step_error}")
                    break

            flow_end_time = time.time()
            total_flow_time = flow_end_time - flow_start_time

            # Calculate statistics
            if model_latencies:
                model_stats = {
                    "total_flow_latency_seconds": round(total_flow_time, 4),
                    "llm_calls_made": len(model_latencies),
                    "individual_call_latencies": [
                        round(lat, 4) for lat in model_latencies
                    ],
                    "step_details": step_details,
                    "statistics": {
                        "mean_call_latency": round(mean(model_latencies), 4),
                        "median_call_latency": round(median(model_latencies), 4),
                        "min_call_latency": round(min(model_latencies), 4),
                        "max_call_latency": round(max(model_latencies), 4),
                        "total_llm_time": round(sum(model_latencies), 4),
                        "overhead_time": round(
                            total_flow_time - sum(model_latencies), 4
                        ),
                    },
                }

                # Add standard deviation if we have multiple calls
                if len(model_latencies) > 1:
                    model_stats["statistics"]["std_dev_latency"] = round(
                        stdev(model_latencies), 4
                    )

                results["model_results"][model_name] = model_stats
                results["test_metadata"]["successful_tests"] += 1

                print(
                    f"  ✅ Success - Total: {total_flow_time:.3f}s, Mean call: {mean(model_latencies):.3f}s"
                )
            else:
                results["model_results"][model_name] = {
                    "error": "No successful LLM calls made",
                    "total_flow_latency_seconds": round(total_flow_time, 4),
                }
                results["test_metadata"]["failed_tests"] += 1

        except Exception as model_error:
            print(f"  ❌ Model {model_name} failed: {model_error}")
            results["model_results"][model_name] = {
                "error": str(model_error),
                "total_flow_latency_seconds": 0,
            }
            results["test_metadata"]["failed_tests"] += 1

    # Generate summary statistics
    successful_results = [
        result
        for result in results["model_results"].values()
        if "error" not in result and "statistics" in result
    ]

    if successful_results:
        all_total_times = [r["total_flow_latency_seconds"] for r in successful_results]
        all_mean_call_times = [
            r["statistics"]["mean_call_latency"] for r in successful_results
        ]

        results["summary"] = {
            "fastest_model": min(
                results["model_results"].items(),
                key=lambda x: x[1].get("total_flow_latency_seconds", float("inf")),
            )[0],
            "slowest_model": max(
                results["model_results"].items(),
                key=lambda x: x[1].get("total_flow_latency_seconds", 0),
            )[0],
            "average_flow_time": round(mean(all_total_times), 4),
            "average_call_time": round(mean(all_mean_call_times), 4),
            "total_test_duration": f"{time.time() - time.time():.2f} seconds",
        }

    # Save results to JSON file
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    except Exception as save_error:
        print(f"⚠️  Could not save to file: {save_error}")

    print(f"\n📈 TESTING COMPLETE")
    print(f"Total models tested: {results['test_metadata']['total_models_tested']}")
    print(f"Successful: {results['test_metadata']['successful_tests']}")
    print(f"Failed: {results['test_metadata']['failed_tests']}")

    if "fastest_model" in results["summary"]:
        print(f"🏆 Fastest model: {results['summary']['fastest_model']}")
        print(f"🐌 Slowest model: {results['summary']['slowest_model']}")

    return results


# Example usage function
def run_latency_test(api_key: str):
    """
    Run the latency test with your API key
    """
    # Test with the default suspicious flow
    suspicious_flow = "Network flow from IoT device 192.168.1.101 to server 10.0.0.2 on port 80 using TCP, duration 0.1 seconds, 1 forward packet, 0 backward packets, 64 bytes total. SYN flag set, no FIN flag."

    results = test_models_latency(
        api_key=api_key,
        test_flow=suspicious_flow,
        output_file="iot_models_latency_test.json",
    )

    return results


if __name__ == "__main__":
    # Initialize IoT SOC agent
    api_key = os.getenv("API_KEY")

    iot_agent = IoTSOCAgent(api_key)

    iot_flows = [
        "Network flow from IoT device 192.168.1.101 to server 10.0.0.2 on port 80 using TCP, duration 0.1 seconds, 1 forward packet, 0 backward packets, 64 bytes total. SYN flag set, no FIN flag.",
        # "MQTT communication from smart thermostat 192.168.1.10 to broker 192.168.1.1 on port 1883, duration 45.2 seconds, 12 forward packets, 11 backward packets, normal IoT sensor data exchange.",
        # "Suspicious activity from 172.16.1.50 to multiple IoT devices on port 23, duration 0.02 seconds, 1 packet each, RST flags, possible Telnet scanning attempt.",
    ]

    run_latency_test(api_key)
