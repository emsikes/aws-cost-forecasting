import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil import parser


# Random seed for reproducability
np.random.seed(42)
random.seed(42)

def generate_cloud_usage_data(n_records=10000):
    """Generate sythetic AWS Cloud usage data"""

    # Define instance types with base hourly costs
    instance_types = {
        't3.micro': 0.0104,
        't3.small': 0.0208,
        't3.medium': 0.0416,
        'm5.large': 0.096,
        'm5.xlarge': 0.192,
        'c5.large': 0.085,
        'c5.xlarge': 0.17,
        'r5.large': 0.126,
        'r5.xlarge': 0.252
    }

    regions = ['us-east-1','us-west-2','eu-west-1', 'ap-southeast-1']
    region_multiplyer = {'us-east-1': 1.0, 'us-west-2': 1.05, 'eu-west-1': 1.1, 'ap-southeast-1': 1.15}

    data = []

    for i in range(n_records):
        # Basic instance info
        instance_type = random.choice(list(instance_types.keys()))
        region = random.choice(regions)

        # Usage patters - some correlation between instance type and usage
        base_cpu = np.random.normal(50, 20)
        if 'c5' in instance_type: # Compute optimized tend to have higher CPU
            base_cpu += 20
        cpu_utilization = np.clip(base_cpu, 5, 95)

        # Memory usage somewhat correlated with CPU
        memory_usage = np.clip(cpu_utilization + np.random.normal(0, 15), 10, 90)

        # Storage usage
        storage_gb = np.random.exponential(100) # Most instances use little storage, smaller percentage use a lot
        storage_gb = np.clip(storage_gb, 10, 1000)

        # Network usage
        network_gb = np.random.exponential(50)
        network_gb = np.clip(network_gb, 1, 500)

        # Usage duration (hours in a month)
        # Some instances run 24/7, others spradic
        if random.random() < 0.3: # 30% always on
            usage_hours = np.random.normal(720, 50) # ~30 days
        else:
            usage_hours = np.random.exponential(200)
        usage_hours = np.clip(usage_hours, 1, 744) # Max hours in a month

        # Time based factors
        month = random.randint(1, 12)
        is_business_hours = random.choice([True, False])

        # Calculate base cost
        base_hourly_cost = instance_types[instance_type] * region_multiplyer[region]

        # Add usage based costs
        compute_cost = base_hourly_cost * usage_hours
        storage_cost = storage_gb * 0.10 # $0.10 per GB per month
        network_cost = network_gb * 0.2 # $0.02 per GB

        # Add in variability and premium pricing during peak hours
        if is_business_hours:
            compute_cost *= 1.2

        # Seasonal pricing variations
        if month in [11, 12, 1]: # Holiday season
            compute_cost *= 1.1

        total_cost = compute_cost + storage_cost + network_cost

        # Add some noise to make it more realistic
        total_cost *= np.random.normal(1.0, 0.05)
        total_cost = max(total_cost, 0.01) # Ensure positive cost

        data.append({
            'instance_type': instance_type,
            'region': region,
            'cpu_utilization_percent': round(cpu_utilization, 2),
            'memory_usage_percent': round(memory_usage, 2),
            'storage_gb': round(storage_gb, 2),
            'network_gb': round(network_gb, 2),
            'usage_hours': round(usage_hours, 2),
            'month': month,
            'is_business_hours': is_business_hours,
            'total_cost': round(total_cost, 2)
        })

    return pd.DataFrame(data)

# Generate the dataset
print("Generating synthetic cloud usage data...")
df = generate_cloud_usage_data(10000)

# Display basic info
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print("\nDataset into:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# Save to CSV
now = datetime.now()
timestamp = now.strftime('%Y-%b-%d-%X')
df.to_csv(f'../datasets/cloud_data_usage_{timestamp}.csv', index=False)
print(f"\nDataset saved as: cloud_data_usage_(current time).csv")

# Show cost distribution
print(f"\nCost Distribution:")
print(f"Mean cost: ${df['total_cost'].median():.2f}")
print(f"Median cost: ${df['total_cost'].median():.2f}")
print(f"Min cost: ${df['total_cost'].min():.2f}")
print(f"Max cost: ${df['total_cost'].max():.2f}")


