from src.pipeline import PromptInjectionPipeline
import argparse

def run_tests(model_dir: str, use_cuda: bool = False):
    print("Initializing Unified Architecture...")
    device = "cuda" if use_cuda else "cpu"
    pipeline = PromptInjectionPipeline(device=device)
    pipeline.load(model_dir)
    
    print("\n--- Model Ready ---")
    print("Type your prompt to evaluate (or 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\n>> Prompt: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if not user_input.strip():
                continue
                
            res = pipeline.analyze_prompt(user_input)
            
            print(f"\n[ DECISION: {res['decision']} ] (Risk Score: {res['risk_score']:.4f})")
            print("--- Breakdown ---")
            for k, v in res['breakdown'].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
                    
        except KeyboardInterrupt:
            break
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive prompt injection detector.")
    parser.add_argument("--model-dir", type=str, default="artifacts/models")
    parser.add_argument("--use-cuda", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()
    run_tests(args.model_dir, args.use_cuda)
