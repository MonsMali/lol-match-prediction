import os
import shutil
from datetime import datetime

def organize_existing_files():
    """
    Organize existing files from the previous run.
    Move visualizations to Visualizations/ directory.
    """
    print("📁 ORGANIZING EXISTING FILES")
    print("=" * 40)
    
    # Create directories if they don't exist
    directories = {
        'Visualizations': 'Visualizations',
        'enhanced_models': 'enhanced_models', 
        'results': 'results'
    }
    
    for dir_name, dir_path in directories.items():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"   ✅ Created: {dir_path}/")
        else:
            print(f"   📁 Exists: {dir_path}/")
    
    # List of visualization files to move
    viz_files = [
        'comprehensive_strategy_comparison.png',
        'pure_temporal_comprehensive_analysis.png',
        'stratified_random_temporal_comprehensive_analysis.png', 
        'stratified_temporal_comprehensive_analysis.png'
    ]
    
    moved_files = []
    
    print(f"\n📊 MOVING VISUALIZATION FILES:")
    
    for viz_file in viz_files:
        if os.path.exists(viz_file):
            # Move to Visualizations directory
            dest_path = os.path.join('Visualizations', viz_file)
            
            # If file already exists in destination, create backup
            if os.path.exists(dest_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{viz_file.replace('.png', '')}_{timestamp}.png"
                backup_path = os.path.join('Visualizations', backup_name)
                shutil.move(dest_path, backup_path)
                print(f"   📦 Backed up existing: {backup_name}")
            
            shutil.move(viz_file, dest_path)
            moved_files.append(viz_file)
            print(f"   ✅ Moved: {viz_file} → Visualizations/{viz_file}")
        else:
            print(f"   ❌ Not found: {viz_file}")
    
    # List of result files to move
    result_files = [
        'comprehensive_comparison_summary.csv',
        'comprehensive_comparison_results.joblib'
    ]
    
    print(f"\n📋 MOVING RESULT FILES:")
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_subdir = os.path.join('results', f'previous_run_{timestamp}')
    os.makedirs(results_subdir, exist_ok=True)
    print(f"   📁 Created: {results_subdir}/")
    
    for result_file in result_files:
        if os.path.exists(result_file):
            dest_path = os.path.join(results_subdir, result_file)
            shutil.move(result_file, dest_path)
            moved_files.append(result_file)
            print(f"   ✅ Moved: {result_file} → {results_subdir}/{result_file}")
        else:
            print(f"   ❌ Not found: {result_file}")
    
    # List of model files to organize
    model_files = [
        'pure_temporal_model.joblib',
        'pure_temporal_scaler.joblib',
        'stratified_temporal_model.joblib', 
        'stratified_temporal_scaler.joblib',
        'stratified_random_temporal_model.joblib',
        'stratified_random_temporal_scaler.joblib'
    ]
    
    print(f"\n🤖 ORGANIZING MODEL FILES:")
    
    for model_file in model_files:
        if os.path.exists(model_file):
            dest_path = os.path.join('enhanced_models', model_file)
            
            # If file already exists, create backup
            if os.path.exists(dest_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{model_file.replace('.joblib', '')}_{timestamp}.joblib"
                backup_path = os.path.join('enhanced_models', backup_name)
                shutil.move(dest_path, backup_path)
                print(f"   📦 Backed up existing: {backup_name}")
            
            shutil.move(model_file, dest_path)
            moved_files.append(model_file)
            print(f"   ✅ Moved: {model_file} → enhanced_models/{model_file}")
        else:
            print(f"   ❌ Not found: {model_file}")
    
    # Clean up any other scattered files
    other_files = [f for f in os.listdir('.') if f.endswith(('.png', '.csv', '.joblib')) and f not in moved_files]
    
    if other_files:
        print(f"\n📄 OTHER FILES FOUND:")
        for file in other_files:
            print(f"   📋 {file} (review manually)")
    
    print(f"\n✅ ORGANIZATION COMPLETE!")
    print(f"   📊 Visualizations: Visualizations/")
    print(f"   🤖 Models: enhanced_models/")
    print(f"   📋 Results: {results_subdir}/")
    print(f"   🗂️ Total files moved: {len(moved_files)}")
    
    # Create a summary file
    summary_path = os.path.join(results_subdir, 'file_organization_log.txt')
    with open(summary_path, 'w') as f:
        f.write(f"File Organization Log\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Files moved: {len(moved_files)}\n\n")
        f.write("Moved files:\n")
        for file in moved_files:
            f.write(f"  - {file}\n")
    
    print(f"   📝 Created log: {summary_path}")
    
    return moved_files

if __name__ == "__main__":
    organize_existing_files() 