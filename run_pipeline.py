"""
Main execution script for the Pharma Context Pipeline.
Run this file to process medicine label images.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config, setup_logging
from src.pipeline import PharmaContextPipeline
from src.utils import save_json, format_output_json

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Pharma Context Engine - Medicine Label Analysis"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image file"
    )
    
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to directory containing multiple images"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(Config.OUTPUT_DIR),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Save intermediate processing results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 70)
    logger.info("Intelligent Pharma Context Engine")
    logger.info("=" * 70)
    
    # Initialize pipeline
    try:
        pipeline = PharmaContextPipeline(Config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        return 1
    
    # Process images
    try:
        if args.image:
            # Single image processing
            logger.info(f"Processing single image: {args.image}")
            result = pipeline.process_image(args.image, save_intermediate=args.intermediate)
            
            # Save result
            output_file = Path(args.output) / "result.json"
            save_json(result, str(output_file))
            
            # Print summary
            print("\n" + "=" * 70)
            print("PROCESSING COMPLETE")
            print("=" * 70)
            print(f"Status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                final = result.get('final_output', {})
                med_info = final.get('medicine_information', {})
                
                print(f"\nDrug Name: {med_info.get('name', 'N/A')}")
                print(f"Manufacturer: {med_info.get('manufacturer', 'N/A')}")
                print(f"Dosage: {med_info.get('dosage', 'N/A')}")
                
                verification = result.get('verification', {})
                print(f"\nVerification Status: {verification.get('verified', False)}")
                print(f"Confidence: {verification.get('confidence', 0):.2%}")
                
                metrics = result.get('metrics', {})
                print(f"\nProcessing Time: {metrics.get('processing_time_seconds', 0):.2f}s")
                print(f"Entity Match Rate: {metrics.get('entity_match_rate', 0):.1f}%")
            
            print(f"\nDetailed results saved to: {output_file}")
            print("=" * 70)
            
        elif args.batch:
            # Batch processing
            logger.info(f"Processing batch from directory: {args.batch}")
            results = pipeline.process_batch(args.batch, args.output)
            
            # Print summary
            successful = sum(1 for r in results if r.get('status') == 'success')
            print("\n" + "=" * 70)
            print("BATCH PROCESSING COMPLETE")
            print("=" * 70)
            print(f"Total Images: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print(f"\nResults saved to: {args.output}")
            print("=" * 70)
            
        else:
            parser.print_help()
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
