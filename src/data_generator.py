import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random

def generate_expanded_insurance_data(original_file="data/insurance.csv", target_size=15000):
    """
    Generate an expanded insurance dataset based on the original data patterns.
    """
    print(f"Loading original data from {original_file}...")
    
    # Load original data
    try:
        original_df = pd.read_csv(original_file)
    except FileNotFoundError:
        print(f"Error: {original_file} not found. Please ensure the file exists.")
        return None
    
    print(f"Original dataset size: {len(original_df)} records")
    print(f"Target size: {target_size} records")
    
    # Analyze original data patterns
    age_stats = original_df['age'].describe()
    bmi_stats = original_df['bmi'].describe()
    charges_stats = original_df['charges'].describe()
    
    # Get categorical distributions
    sex_dist = original_df['sex'].value_counts(normalize=True)
    smoker_dist = original_df['smoker'].value_counts(normalize=True)
    region_dist = original_df['region'].value_counts(normalize=True)
    children_dist = original_df['children'].value_counts(normalize=True)
    
    print("Generating expanded dataset...")
    
    # Generate new data
    new_data = []
    
    for i in range(target_size):
        # Generate age (normal distribution around mean)
        age = int(np.random.normal(age_stats['mean'], age_stats['std']))
        age = max(18, min(100, age))  # Clamp to realistic range
        
        # Generate sex
        sex = np.random.choice(sex_dist.index, p=sex_dist.values)
        
        # Generate BMI (normal distribution)
        bmi = np.random.normal(bmi_stats['mean'], bmi_stats['std'])
        bmi = max(15, min(50, bmi))  # Clamp to realistic range
        
        # Generate children (poisson distribution)
        children = np.random.poisson(children_dist.index[children_dist.argmax()])
        children = min(children, 8)  # Cap at 8 children
        
        # Generate smoker status
        smoker = np.random.choice(smoker_dist.index, p=smoker_dist.values)
        
        # Generate region
        region = np.random.choice(region_dist.index, p=region_dist.values)
        
        # Generate charges based on features (with some randomness)
        base_charge = 1000 + age * 50 + bmi * 100
        
        # Smoker multiplier
        if smoker == 'yes':
            base_charge *= 3.5
        
        # Children effect
        base_charge += children * 500
        
        # Region effect
        region_multipliers = {
            'northeast': 1.1,
            'northwest': 1.0,
            'southeast': 0.9,
            'southwest': 0.95
        }
        base_charge *= region_multipliers.get(region, 1.0)
        
        # Add some randomness
        charge = base_charge * np.random.uniform(0.7, 1.3)
        charge = max(1000, charge)  # Minimum charge
        
        new_data.append({
            'age': age,
            'sex': sex,
            'bmi': round(bmi, 2),
            'children': children,
            'smoker': smoker,
            'region': region,
            'charges': round(charge, 2)
        })
    
    # Create expanded dataset
    expanded_df = pd.DataFrame(new_data)
    
    # Save expanded dataset
    output_file = "data/insurance_expanded.csv"
    expanded_df.to_csv(output_file, index=False)
    
    print(f"Generated {len(expanded_df)} records")
    print(f"Saved to {output_file}")
    
    # Print summary statistics
    print("\nExpanded Dataset Summary:")
    print(f"Age range: {expanded_df['age'].min()} - {expanded_df['age'].max()}")
    print(f"BMI range: {expanded_df['bmi'].min():.2f} - {expanded_df['bmi'].max():.2f}")
    print(f"Charges range: ${expanded_df['charges'].min():,.2f} - ${expanded_df['charges'].max():,.2f}")
    print(f"Smoker percentage: {(expanded_df['smoker'] == 'yes').mean()*100:.1f}%")
    
    return expanded_df

if __name__ == "__main__":
    # Generate expanded dataset
    expanded_data = generate_expanded_insurance_data()
    
    if expanded_data is not None:
        print("\nDataset generation completed successfully!")
        print("You can now use 'data/insurance_expanded.csv' for benchmarking experiments.") 