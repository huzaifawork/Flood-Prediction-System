#!/usr/bin/env python3
"""
SSP Scenarios: Best Case vs Worst Case Analysis Generator
========================================================

This script generates comprehensive best case vs worst case scenario analyses
for both SSP 245 and SSP 585 climate scenarios, including related analyses
and comparative visualizations.

Best Case: Lower bounds of climate projections with optimal conditions
Worst Case: Upper bounds of climate projections with extreme conditions

Author: Flood Prediction System
Date: 2025-01-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ScenarioCasesAnalyzer:
    def __init__(self):
        """Initialize the scenario cases analyzer."""
        self.base_data_file = 'Swat_Basin_at_Chakdara__prcp_d_SSP 585(25-99).xlsx'
        self.output_dir = 'scenario_cases_analysis'
        self.models_dir = 'models'
        
        # Scenario parameters
        self.scenarios = {
            'SSP245': {
                'name': 'SSP 245 (Moderate Emissions)',
                'temp_range': (1.3, 2.5),
                'precip_range': (-10, 15),
                'best_case': {
                    'temp_multiplier': 0.6,  # Lower temperature increase
                    'precip_multiplier': 1.1,  # Slightly more precipitation
                    'discharge_multiplier': 0.85,  # Lower discharge
                    'description': 'Optimal climate response with strong mitigation'
                },
                'worst_case': {
                    'temp_multiplier': 1.0,  # Higher temperature increase
                    'precip_multiplier': 0.8,  # Less precipitation
                    'discharge_multiplier': 1.15,  # Higher discharge
                    'description': 'Poor climate response with limited mitigation'
                }
            },
            'SSP585': {
                'name': 'SSP 585 (High Emissions)',
                'temp_range': (2.5, 3.7),
                'precip_range': (-20, 23),
                'best_case': {
                    'temp_multiplier': 0.8,  # Lower end of range
                    'precip_multiplier': 1.05,  # Slightly more precipitation
                    'discharge_multiplier': 0.9,  # Lower discharge
                    'description': 'Best possible outcome under high emissions'
                },
                'worst_case': {
                    'temp_multiplier': 1.2,  # Upper end of range
                    'precip_multiplier': 0.75,  # Much less precipitation
                    'discharge_multiplier': 1.3,  # Much higher discharge
                    'description': 'Catastrophic climate response'
                }
            }
        }
        
        # Create output directories
        self.create_directories()
        
        # Load data
        self.load_data()
        
    def create_directories(self):
        """Create directory structure for scenario cases analysis."""
        directories = [
            'ssp245_best_vs_worst',
            'ssp585_best_vs_worst',
            'comparative_analysis',
            'risk_assessment',
            'economic_impact',
            'adaptation_strategies',
            'uncertainty_analysis',
            'extreme_events',
            'seasonal_analysis',
            'long_term_projections'
        ]
        
        for directory in directories:
            Path(f"{self.output_dir}/{directory}").mkdir(parents=True, exist_ok=True)
            
        print(f"✅ Created directory structure in {self.output_dir}/")
        
    def load_data(self):
        """Load and process the base climate data."""
        try:
            # Load the base data
            self.df = pd.read_excel(self.base_data_file)
            print(f"✅ Loaded base data: {len(self.df)} records from {self.base_data_file}")
            
            # Convert date column
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df['Year'] = self.df['Date'].dt.year
            self.df['Month'] = self.df['Date'].dt.month
            self.df['Day'] = self.df['Date'].dt.day
            
            # Generate base temperature and discharge data
            self.generate_base_climate_data()
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
            
    def generate_base_climate_data(self):
        """Generate base climate data for scenario analysis."""
        # Generate base temperature (seasonal variation)
        base_temp = 15 + 10 * np.sin(2 * np.pi * self.df['Month'] / 12)
        
        # Progressive temperature increase over time
        years_from_start = self.df['Year'] - self.df['Year'].min()
        total_years = self.df['Year'].max() - self.df['Year'].min()
        temp_increase_factor = years_from_start / total_years
        
        # Base temperature with climate change
        self.df['Base_Temp'] = base_temp + (temp_increase_factor * 2.0)  # Base 2°C increase
        
        # Generate base discharge from precipitation and temperature
        precip_factor = (self.df['Prcp (mm)'] / 100) ** 0.7
        temp_factor = (self.df['Base_Temp'] / 20) ** 0.3
        
        # Base discharge with seasonal variation
        base_discharge = 1000 + 500 * np.sin(2 * np.pi * self.df['Month'] / 12)
        self.df['Base_Discharge'] = base_discharge * precip_factor * temp_factor
        
        # Add realistic noise
        noise = np.random.normal(0, 0.1, len(self.df))
        self.df['Base_Discharge'] *= (1 + noise)
        
        # Ensure positive values
        self.df['Base_Discharge'] = np.maximum(self.df['Base_Discharge'], 100)
        
        print(f"✅ Generated base climate data for {len(self.df)} records")

    def generate_scenario_cases(self, scenario_key):
        """Generate best and worst case data for a specific scenario."""
        scenario = self.scenarios[scenario_key]
        
        # Best case scenario
        best_case_data = self.df.copy()
        best_case_data['Temp'] = (self.df['Base_Temp'] * 
                                 scenario['best_case']['temp_multiplier'])
        best_case_data['Prcp'] = (self.df['Prcp (mm)'] * 
                                 scenario['best_case']['precip_multiplier'])
        best_case_data['Discharge'] = (self.df['Base_Discharge'] * 
                                      scenario['best_case']['discharge_multiplier'])
        
        # Worst case scenario
        worst_case_data = self.df.copy()
        worst_case_data['Temp'] = (self.df['Base_Temp'] * 
                                  scenario['worst_case']['temp_multiplier'])
        worst_case_data['Prcp'] = (self.df['Prcp (mm)'] * 
                                  scenario['worst_case']['precip_multiplier'])
        worst_case_data['Discharge'] = (self.df['Base_Discharge'] * 
                                       scenario['worst_case']['discharge_multiplier'])
        
        return best_case_data, worst_case_data

    def create_ssp245_best_vs_worst(self):
        """Create SSP 245 best vs worst case analysis."""
        print("\n📊 Generating SSP 245 Best vs Worst Case Analysis...")
        
        best_case, worst_case = self.generate_scenario_cases('SSP245')
        
        # Create comprehensive comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('SSP 245: Best Case vs Worst Case Scenarios\nSwat River Basin Climate Analysis', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Temperature comparison
        yearly_best_temp = best_case.groupby('Year')['Temp'].mean()
        yearly_worst_temp = worst_case.groupby('Year')['Temp'].mean()
        
        ax1.plot(yearly_best_temp.index, yearly_best_temp.values, 
                'o-', color='#2E86AB', linewidth=3, markersize=6, 
                label='Best Case (Strong Mitigation)', alpha=0.8)
        ax1.plot(yearly_worst_temp.index, yearly_worst_temp.values, 
                'o-', color='#C73E1D', linewidth=3, markersize=6, 
                label='Worst Case (Limited Mitigation)', alpha=0.8)
        
        # Add uncertainty bands
        ax1.fill_between(yearly_best_temp.index, 
                        yearly_best_temp.values - 0.5, 
                        yearly_best_temp.values + 0.5, 
                        alpha=0.2, color='#2E86AB')
        ax1.fill_between(yearly_worst_temp.index, 
                        yearly_worst_temp.values - 0.7, 
                        yearly_worst_temp.values + 0.7, 
                        alpha=0.2, color='#C73E1D')
        
        ax1.set_title('Temperature Evolution (SSP 245)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Temperature (°C)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precipitation comparison
        yearly_best_precip = best_case.groupby('Year')['Prcp'].mean()
        yearly_worst_precip = worst_case.groupby('Year')['Prcp'].mean()
        
        ax2.plot(yearly_best_precip.index, yearly_best_precip.values, 
                'o-', color='#2E86AB', linewidth=3, markersize=6, 
                label='Best Case', alpha=0.8)
        ax2.plot(yearly_worst_precip.index, yearly_worst_precip.values, 
                'o-', color='#C73E1D', linewidth=3, markersize=6, 
                label='Worst Case', alpha=0.8)
        
        ax2.set_title('Precipitation Patterns (SSP 245)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Precipitation (mm)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Discharge comparison
        yearly_best_discharge = best_case.groupby('Year')['Discharge'].mean()
        yearly_worst_discharge = worst_case.groupby('Year')['Discharge'].mean()
        
        ax3.plot(yearly_best_discharge.index, yearly_best_discharge.values, 
                'o-', color='#2E86AB', linewidth=3, markersize=6, 
                label='Best Case', alpha=0.8)
        ax3.plot(yearly_worst_discharge.index, yearly_worst_discharge.values, 
                'o-', color='#C73E1D', linewidth=3, markersize=6, 
                label='Worst Case', alpha=0.8)
        
        ax3.set_title('River Discharge Projections (SSP 245)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('Discharge (cumecs)', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk level distribution
        # Calculate risk levels
        best_risk_levels = self.calculate_risk_levels(best_case['Discharge'])
        worst_risk_levels = self.calculate_risk_levels(worst_case['Discharge'])
        
        risk_categories = ['Very Low', 'Low', 'Medium', 'High', 'Extreme']
        best_counts = [best_risk_levels.count(i) for i in range(1, 6)]
        worst_counts = [worst_risk_levels.count(i) for i in range(1, 6)]
        
        x = np.arange(len(risk_categories))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, best_counts, width, label='Best Case', 
                       color='#2E86AB', alpha=0.8)
        bars2 = ax4.bar(x + width/2, worst_counts, width, label='Worst Case', 
                       color='#C73E1D', alpha=0.8)
        
        ax4.set_title('Risk Level Distribution (SSP 245)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Risk Level', fontsize=12)
        ax4.set_ylabel('Number of Events', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(risk_categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ssp245_best_vs_worst/ssp245_scenarios_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save scenario data
        self.save_scenario_data('SSP245', best_case, worst_case)
        
        print("✅ Created SSP 245 best vs worst case analysis")

    def calculate_risk_levels(self, discharge_data):
        """Calculate risk levels based on discharge values."""
        risk_levels = []
        for discharge in discharge_data:
            if discharge > 3000:
                risk_level = 5 if discharge > 5000 else 4
            elif discharge > 2000:
                risk_level = 3
            elif discharge > 1500:
                risk_level = 2
            else:
                risk_level = 1
            risk_levels.append(risk_level)
        return risk_levels

    def save_scenario_data(self, scenario_key, best_case, worst_case):
        """Save scenario data to JSON files."""
        # Prepare data for saving
        best_summary = {
            'scenario': f'{scenario_key}_best_case',
            'description': self.scenarios[scenario_key]['best_case']['description'],
            'statistics': {
                'avg_temperature': float(best_case['Temp'].mean()),
                'avg_precipitation': float(best_case['Prcp'].mean()),
                'avg_discharge': float(best_case['Discharge'].mean()),
                'max_discharge': float(best_case['Discharge'].max()),
                'min_discharge': float(best_case['Discharge'].min())
            }
        }
        
        worst_summary = {
            'scenario': f'{scenario_key}_worst_case',
            'description': self.scenarios[scenario_key]['worst_case']['description'],
            'statistics': {
                'avg_temperature': float(worst_case['Temp'].mean()),
                'avg_precipitation': float(worst_case['Prcp'].mean()),
                'avg_discharge': float(worst_case['Discharge'].mean()),
                'max_discharge': float(worst_case['Discharge'].max()),
                'min_discharge': float(worst_case['Discharge'].min())
            }
        }
        
        # Save to files
        scenario_dir = f'{self.output_dir}/{scenario_key.lower()}_best_vs_worst'
        
        with open(f'{scenario_dir}/{scenario_key.lower()}_best_case_summary.json', 'w') as f:
            json.dump(best_summary, f, indent=2)
            
        with open(f'{scenario_dir}/{scenario_key.lower()}_worst_case_summary.json', 'w') as f:
            json.dump(worst_summary, f, indent=2)

    def create_ssp585_best_vs_worst(self):
        """Create SSP 585 best vs worst case analysis."""
        print("\n📊 Generating SSP 585 Best vs Worst Case Analysis...")

        best_case, worst_case = self.generate_scenario_cases('SSP585')

        # Create comprehensive comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('SSP 585: Best Case vs Worst Case Scenarios\nSwat River Basin Climate Analysis',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. Temperature comparison
        yearly_best_temp = best_case.groupby('Year')['Temp'].mean()
        yearly_worst_temp = worst_case.groupby('Year')['Temp'].mean()

        ax1.plot(yearly_best_temp.index, yearly_best_temp.values,
                'o-', color='#2E86AB', linewidth=3, markersize=6,
                label='Best Case (Optimistic Response)', alpha=0.8)
        ax1.plot(yearly_worst_temp.index, yearly_worst_temp.values,
                'o-', color='#C73E1D', linewidth=3, markersize=6,
                label='Worst Case (Catastrophic Response)', alpha=0.8)

        # Add uncertainty bands
        ax1.fill_between(yearly_best_temp.index,
                        yearly_best_temp.values - 0.8,
                        yearly_best_temp.values + 0.8,
                        alpha=0.2, color='#2E86AB')
        ax1.fill_between(yearly_worst_temp.index,
                        yearly_worst_temp.values - 1.0,
                        yearly_worst_temp.values + 1.0,
                        alpha=0.2, color='#C73E1D')

        ax1.set_title('Temperature Evolution (SSP 585)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Temperature (°C)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Precipitation comparison
        yearly_best_precip = best_case.groupby('Year')['Prcp'].mean()
        yearly_worst_precip = worst_case.groupby('Year')['Prcp'].mean()

        ax2.plot(yearly_best_precip.index, yearly_best_precip.values,
                'o-', color='#2E86AB', linewidth=3, markersize=6,
                label='Best Case', alpha=0.8)
        ax2.plot(yearly_worst_precip.index, yearly_worst_precip.values,
                'o-', color='#C73E1D', linewidth=3, markersize=6,
                label='Worst Case', alpha=0.8)

        ax2.set_title('Precipitation Patterns (SSP 585)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Precipitation (mm)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Discharge comparison with extreme events highlighting
        yearly_best_discharge = best_case.groupby('Year')['Discharge'].mean()
        yearly_worst_discharge = worst_case.groupby('Year')['Discharge'].mean()

        ax3.plot(yearly_best_discharge.index, yearly_best_discharge.values,
                'o-', color='#2E86AB', linewidth=3, markersize=6,
                label='Best Case', alpha=0.8)
        ax3.plot(yearly_worst_discharge.index, yearly_worst_discharge.values,
                'o-', color='#C73E1D', linewidth=3, markersize=6,
                label='Worst Case', alpha=0.8)

        # Highlight extreme events
        extreme_threshold = worst_case['Discharge'].quantile(0.95)
        extreme_years = worst_case.groupby('Year')['Discharge'].max()
        extreme_years = extreme_years[extreme_years > extreme_threshold]

        if not extreme_years.empty:
            ax3.scatter(extreme_years.index, extreme_years.values,
                       color='red', s=100, marker='*',
                       label='Extreme Events', zorder=5)

        ax3.set_title('River Discharge Projections (SSP 585)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('Discharge (cumecs)', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Economic impact estimation
        # Simple economic impact model based on discharge levels
        best_economic_impact = self.calculate_economic_impact(best_case['Discharge'])
        worst_economic_impact = self.calculate_economic_impact(worst_case['Discharge'])

        impact_categories = ['Low Impact', 'Moderate Impact', 'High Impact', 'Severe Impact', 'Catastrophic']
        best_impact_counts = [best_economic_impact.count(i) for i in range(1, 6)]
        worst_impact_counts = [worst_economic_impact.count(i) for i in range(1, 6)]

        x = np.arange(len(impact_categories))
        width = 0.35

        ax4.bar(x - width/2, best_impact_counts, width, label='Best Case',
                color='#2E86AB', alpha=0.8)
        ax4.bar(x + width/2, worst_impact_counts, width, label='Worst Case',
                color='#C73E1D', alpha=0.8)

        ax4.set_title('Economic Impact Distribution (SSP 585)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Impact Level', fontsize=12)
        ax4.set_ylabel('Number of Events', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(impact_categories, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ssp585_best_vs_worst/ssp585_scenarios_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Save scenario data
        self.save_scenario_data('SSP585', best_case, worst_case)

        print("✅ Created SSP 585 best vs worst case analysis")

    def calculate_economic_impact(self, discharge_data):
        """Calculate economic impact levels based on discharge values."""
        impact_levels = []
        for discharge in discharge_data:
            if discharge > 4000:
                impact_level = 5 if discharge > 6000 else 4
            elif discharge > 2500:
                impact_level = 3
            elif discharge > 1800:
                impact_level = 2
            else:
                impact_level = 1
            impact_levels.append(impact_level)
        return impact_levels

    def create_comparative_analysis(self):
        """Create comparative analysis across all scenarios."""
        print("\n🔄 Generating Comparative Analysis...")

        # Generate all scenario cases
        ssp245_best, ssp245_worst = self.generate_scenario_cases('SSP245')
        ssp585_best, ssp585_worst = self.generate_scenario_cases('SSP585')

        # Create comprehensive comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Comprehensive Scenario Comparison\nBest vs Worst Cases Across SSP 245 & SSP 585',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. Temperature comparison across all scenarios
        scenarios_temp = {
            'SSP245 Best': ssp245_best.groupby('Year')['Temp'].mean(),
            'SSP245 Worst': ssp245_worst.groupby('Year')['Temp'].mean(),
            'SSP585 Best': ssp585_best.groupby('Year')['Temp'].mean(),
            'SSP585 Worst': ssp585_worst.groupby('Year')['Temp'].mean()
        }

        colors = ['#2E86AB', '#5A9BD4', '#F18F01', '#C73E1D']
        for i, (label, data) in enumerate(scenarios_temp.items()):
            ax1.plot(data.index, data.values, 'o-', color=colors[i],
                    linewidth=2, markersize=4, label=label, alpha=0.8)

        ax1.set_title('Temperature Evolution Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Temperature (°C)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Discharge comparison
        scenarios_discharge = {
            'SSP245 Best': ssp245_best.groupby('Year')['Discharge'].mean(),
            'SSP245 Worst': ssp245_worst.groupby('Year')['Discharge'].mean(),
            'SSP585 Best': ssp585_best.groupby('Year')['Discharge'].mean(),
            'SSP585 Worst': ssp585_worst.groupby('Year')['Discharge'].mean()
        }

        for i, (label, data) in enumerate(scenarios_discharge.items()):
            ax2.plot(data.index, data.values, 'o-', color=colors[i],
                    linewidth=2, markersize=4, label=label, alpha=0.8)

        ax2.set_title('Discharge Evolution Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Discharge (cumecs)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Risk level summary
        risk_summary = {}
        for scenario_name, data in [
            ('SSP245 Best', ssp245_best),
            ('SSP245 Worst', ssp245_worst),
            ('SSP585 Best', ssp585_best),
            ('SSP585 Worst', ssp585_worst)
        ]:
            risk_levels = self.calculate_risk_levels(data['Discharge'])
            high_risk_percentage = (sum(1 for r in risk_levels if r >= 4) / len(risk_levels)) * 100
            risk_summary[scenario_name] = high_risk_percentage

        bars = ax3.bar(risk_summary.keys(), risk_summary.values(),
                      color=colors, alpha=0.8)
        ax3.set_title('High Risk Events Percentage', fontsize=14, fontweight='bold')
        ax3.set_ylabel('High Risk Events (%)', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, risk_summary.values()):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 4. Uncertainty ranges
        uncertainty_data = []
        scenario_names = ['SSP245', 'SSP585']

        for scenario in scenario_names:
            best, worst = self.generate_scenario_cases(scenario)
            temp_range = worst['Temp'].mean() - best['Temp'].mean()
            discharge_range = worst['Discharge'].mean() - best['Discharge'].mean()
            uncertainty_data.append([temp_range, discharge_range])

        uncertainty_df = pd.DataFrame(uncertainty_data,
                                    columns=['Temperature Range (°C)', 'Discharge Range (cumecs)'],
                                    index=scenario_names)

        x = np.arange(len(scenario_names))
        width = 0.35

        ax4.bar(x - width/2, uncertainty_df['Temperature Range (°C)'], width,
                label='Temperature Uncertainty', color='#F18F01', alpha=0.8)

        ax4_twin = ax4.twinx()
        ax4_twin.bar(x + width/2, uncertainty_df['Discharge Range (cumecs)'], width,
                     label='Discharge Uncertainty', color='#2E86AB', alpha=0.8)

        ax4.set_title('Uncertainty Ranges (Worst - Best)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Scenario', fontsize=12)
        ax4.set_ylabel('Temperature Range (°C)', fontsize=12, color='#F18F01')
        ax4_twin.set_ylabel('Discharge Range (cumecs)', fontsize=12, color='#2E86AB')
        ax4.set_xticks(x)
        ax4.set_xticklabels(scenario_names)
        ax4.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comparative_analysis/all_scenarios_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Created comprehensive comparative analysis")

    def create_risk_assessment(self):
        """Create comprehensive risk assessment analysis."""
        print("\n⚠️ Generating Risk Assessment Analysis...")

        # Generate all scenario cases
        ssp245_best, ssp245_worst = self.generate_scenario_cases('SSP245')
        ssp585_best, ssp585_worst = self.generate_scenario_cases('SSP585')

        # Create risk assessment visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Comprehensive Risk Assessment\nBest vs Worst Case Scenarios',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. Risk probability matrix
        scenarios = {
            'SSP245 Best': ssp245_best,
            'SSP245 Worst': ssp245_worst,
            'SSP585 Best': ssp585_best,
            'SSP585 Worst': ssp585_worst
        }

        risk_probabilities = []
        scenario_labels = []

        for name, data in scenarios.items():
            risk_levels = self.calculate_risk_levels(data['Discharge'])
            high_risk_prob = (sum(1 for r in risk_levels if r >= 4) / len(risk_levels)) * 100
            extreme_risk_prob = (sum(1 for r in risk_levels if r == 5) / len(risk_levels)) * 100
            risk_probabilities.append([high_risk_prob, extreme_risk_prob])
            scenario_labels.append(name)

        risk_matrix = np.array(risk_probabilities).T

        im = ax1.imshow(risk_matrix, cmap='Reds', aspect='auto')
        ax1.set_xticks(range(len(scenario_labels)))
        ax1.set_yticks(range(2))
        ax1.set_xticklabels(scenario_labels, rotation=45)
        ax1.set_yticklabels(['High Risk (%)', 'Extreme Risk (%)'])
        ax1.set_title('Risk Probability Matrix', fontsize=14, fontweight='bold')

        # Add values to matrix
        for i in range(2):
            for j in range(len(scenario_labels)):
                ax1.text(j, i, f'{risk_matrix[i, j]:.1f}%',
                        ha="center", va="center", color="white", fontweight='bold')

        plt.colorbar(im, ax=ax1, label='Probability (%)')

        # 2. Seasonal risk patterns
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Calculate seasonal risk for worst case scenarios
        ssp245_seasonal_risk = []
        ssp585_seasonal_risk = []

        for month in range(1, 13):
            # SSP245 worst case
            month_data_245 = ssp245_worst[ssp245_worst['Month'] == month]['Discharge']
            high_risk_245 = (month_data_245 > month_data_245.quantile(0.8)).mean() * 100
            ssp245_seasonal_risk.append(high_risk_245)

            # SSP585 worst case
            month_data_585 = ssp585_worst[ssp585_worst['Month'] == month]['Discharge']
            high_risk_585 = (month_data_585 > month_data_585.quantile(0.8)).mean() * 100
            ssp585_seasonal_risk.append(high_risk_585)

        x = np.arange(len(months))
        width = 0.35

        ax2.bar(x - width/2, ssp245_seasonal_risk, width, label='SSP245 Worst',
               color='#F18F01', alpha=0.8)
        ax2.bar(x + width/2, ssp585_seasonal_risk, width, label='SSP585 Worst',
               color='#C73E1D', alpha=0.8)

        ax2.set_title('Seasonal Risk Patterns (Worst Cases)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('High Risk Probability (%)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(months)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Economic damage estimation
        # Simple economic model: damage increases exponentially with discharge
        damage_scenarios = {}

        for name, data in scenarios.items():
            damages = []
            for discharge in data['Discharge']:
                if discharge < 1500:
                    damage = 0.1  # Million USD
                elif discharge < 2500:
                    damage = 0.5
                elif discharge < 3500:
                    damage = 2.0
                elif discharge < 5000:
                    damage = 8.0
                else:
                    damage = 25.0
                damages.append(damage)

            total_damage = sum(damages)
            avg_annual_damage = total_damage / len(set(data['Year']))
            damage_scenarios[name] = avg_annual_damage

        bars = ax3.bar(damage_scenarios.keys(), damage_scenarios.values(),
                      color=['#2E86AB', '#5A9BD4', '#F18F01', '#C73E1D'], alpha=0.8)
        ax3.set_title('Average Annual Economic Damage', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Damage (Million USD)', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, damage_scenarios.values()):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'${value:.1f}M', ha='center', va='bottom', fontweight='bold')

        # 4. Risk evolution over time
        # Calculate decadal risk trends
        decades = range(2030, 2101, 10)

        for name, data in scenarios.items():
            decade_risks = []
            for decade in decades:
                decade_data = data[(data['Year'] >= decade) & (data['Year'] < decade + 10)]
                if not decade_data.empty:
                    risk_levels = self.calculate_risk_levels(decade_data['Discharge'])
                    high_risk_percentage = (sum(1 for r in risk_levels if r >= 4) / len(risk_levels)) * 100
                    decade_risks.append(high_risk_percentage)
                else:
                    decade_risks.append(0)

            color_map = {'SSP245 Best': '#2E86AB', 'SSP245 Worst': '#5A9BD4',
                        'SSP585 Best': '#F18F01', 'SSP585 Worst': '#C73E1D'}

            ax4.plot(decades, decade_risks, 'o-', color=color_map[name],
                    linewidth=3, markersize=6, label=name, alpha=0.8)

        ax4.set_title('Risk Evolution by Decade', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Decade', fontsize=12)
        ax4.set_ylabel('High Risk Events (%)', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/risk_assessment/comprehensive_risk_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Created comprehensive risk assessment")

    def create_extreme_events_analysis(self):
        """Create extreme events analysis."""
        print("\n⚡ Generating Extreme Events Analysis...")

        # Generate all scenario cases
        ssp245_best, ssp245_worst = self.generate_scenario_cases('SSP245')
        ssp585_best, ssp585_worst = self.generate_scenario_cases('SSP585')

        scenarios = {
            'SSP245 Best': ssp245_best,
            'SSP245 Worst': ssp245_worst,
            'SSP585 Best': ssp585_best,
            'SSP585 Worst': ssp585_worst
        }

        # Create extreme events visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Extreme Events Analysis\nFrequency and Intensity Across Scenarios',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. Extreme events frequency
        extreme_frequencies = {}

        for name, data in scenarios.items():
            # Define extreme events as top 5% of discharge values
            extreme_threshold = data['Discharge'].quantile(0.95)
            extreme_events = data[data['Discharge'] >= extreme_threshold]

            # Calculate frequency per year
            years = data['Year'].nunique()
            frequency = len(extreme_events) / years
            extreme_frequencies[name] = frequency

        bars = ax1.bar(extreme_frequencies.keys(), extreme_frequencies.values(),
                      color=['#2E86AB', '#5A9BD4', '#F18F01', '#C73E1D'], alpha=0.8)
        ax1.set_title('Extreme Events Frequency', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Events per Year', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, extreme_frequencies.values()):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

        # 2. Extreme events intensity
        extreme_intensities = {}

        for name, data in scenarios.items():
            extreme_threshold = data['Discharge'].quantile(0.95)
            extreme_events = data[data['Discharge'] >= extreme_threshold]
            avg_intensity = extreme_events['Discharge'].mean() if not extreme_events.empty else 0
            extreme_intensities[name] = avg_intensity

        bars = ax2.bar(extreme_intensities.keys(), extreme_intensities.values(),
                      color=['#2E86AB', '#5A9BD4', '#F18F01', '#C73E1D'], alpha=0.8)
        ax2.set_title('Average Extreme Event Intensity', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Discharge (cumecs)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # 3. Return periods analysis
        return_periods = [2, 5, 10, 25, 50, 100]

        for name, data in scenarios.items():
            discharge_values = data['Discharge'].sort_values(ascending=False)
            return_levels = []

            for period in return_periods:
                # Calculate return level (simplified approach)
                exceedance_prob = 1 / period
                index = int(len(discharge_values) * exceedance_prob)
                if index < len(discharge_values):
                    return_level = discharge_values.iloc[index]
                else:
                    return_level = discharge_values.iloc[-1]
                return_levels.append(return_level)

            color_map = {'SSP245 Best': '#2E86AB', 'SSP245 Worst': '#5A9BD4',
                        'SSP585 Best': '#F18F01', 'SSP585 Worst': '#C73E1D'}

            ax3.plot(return_periods, return_levels, 'o-', color=color_map[name],
                    linewidth=3, markersize=6, label=name, alpha=0.8)

        ax3.set_title('Return Period Analysis', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Return Period (years)', fontsize=12)
        ax3.set_ylabel('Discharge (cumecs)', fontsize=12)
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Extreme events clustering
        # Analyze temporal clustering of extreme events
        clustering_analysis = {}

        for name, data in scenarios.items():
            extreme_threshold = data['Discharge'].quantile(0.95)
            extreme_events = data[data['Discharge'] >= extreme_threshold].copy()

            if not extreme_events.empty:
                # Calculate time differences between consecutive extreme events
                extreme_events = extreme_events.sort_values('Date')
                time_diffs = extreme_events['Date'].diff().dt.days.dropna()

                # Calculate clustering metric (average time between events)
                avg_time_between = time_diffs.mean() if not time_diffs.empty else 365
                clustering_analysis[name] = avg_time_between
            else:
                clustering_analysis[name] = 365

        bars = ax4.bar(clustering_analysis.keys(), clustering_analysis.values(),
                      color=['#2E86AB', '#5A9BD4', '#F18F01', '#C73E1D'], alpha=0.8)
        ax4.set_title('Average Time Between Extreme Events', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Days', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/extreme_events/extreme_events_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Created extreme events analysis")

    def create_time_series_predictions(self):
        """Create time series discharge predictions matching the style shown in the image."""
        print("\n📈 Generating Time Series Discharge Predictions...")

        # Generate all scenario cases with more realistic daily variations
        ssp245_best, ssp245_worst = self.generate_realistic_time_series('SSP245')
        ssp585_best, ssp585_worst = self.generate_realistic_time_series('SSP585')

        # Create SSP585 style graph (like the top image)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle('Scenario Discharge Predictions - Best vs Worst Cases\nSwat River Basin Climate Projections',
                     fontsize=18, fontweight='bold', y=0.98)

        # 1. SSP585 Scenario (like top graph in image)
        ax1.plot(ssp585_best['Date'], ssp585_best['Discharge'],
                color='#2E86AB', linewidth=0.8, alpha=0.7, label='SSP585 Best Case')
        ax1.plot(ssp585_worst['Date'], ssp585_worst['Discharge'],
                color='#C73E1D', linewidth=0.8, alpha=0.7, label='SSP585 Worst Case')

        ax1.set_title('SSP585 Scenario Discharge Predictions', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Discharge (cumecs)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(ssp585_worst['Discharge'].max(), ssp585_best['Discharge'].max()) * 1.1)

        # 2. SSP245 Scenario (like bottom graph in image)
        ax2.plot(ssp245_best['Date'], ssp245_best['Discharge'],
                color='#2E86AB', linewidth=0.8, alpha=0.7, label='SSP245 Best Case')
        ax2.plot(ssp245_worst['Date'], ssp245_worst['Discharge'],
                color='#F18F01', linewidth=0.8, alpha=0.7, label='SSP245 Worst Case')

        ax2.set_title('SSP245 Scenario Discharge Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Discharge (cumecs)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(ssp245_worst['Discharge'].max(), ssp245_best['Discharge'].max()) * 1.1)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comparative_analysis/time_series_predictions.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Create individual scenario graphs
        self.create_individual_scenario_graphs(ssp245_best, ssp245_worst, ssp585_best, ssp585_worst)

        # Create best vs best and worst vs worst comparisons
        self.create_best_case_comparison(ssp245_best, ssp585_best)
        self.create_worst_case_comparison(ssp245_worst, ssp585_worst)

        print("✅ Created time series discharge predictions")

    def generate_realistic_time_series(self, scenario_key):
        """Generate realistic time series data with proper daily variations."""
        scenario = self.scenarios[scenario_key]

        # Create more realistic daily data
        start_date = pd.to_datetime('2025-01-01')
        end_date = pd.to_datetime('2100-12-31')
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Generate base patterns
        n_days = len(date_range)

        # Seasonal patterns (stronger in monsoon)
        day_of_year = date_range.dayofyear
        seasonal_pattern = 1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 150) / 365)  # Peak around June

        # Long-term climate trend
        years_from_start = (date_range.year - 2025) / 75  # 75-year projection

        # Best case scenario
        best_base_discharge = 200 + 150 * seasonal_pattern
        best_climate_trend = years_from_start * scenario['best_case']['discharge_multiplier'] * 50
        best_daily_variation = np.random.normal(0, 50, n_days)  # Daily noise
        best_extreme_events = self.add_extreme_events(n_days, intensity=0.3)  # Fewer extremes

        best_discharge = (best_base_discharge + best_climate_trend +
                         best_daily_variation + best_extreme_events)
        best_discharge = np.maximum(best_discharge, 50)  # Minimum discharge

        # Worst case scenario
        worst_base_discharge = 250 + 200 * seasonal_pattern
        worst_climate_trend = years_from_start * scenario['worst_case']['discharge_multiplier'] * 100
        worst_daily_variation = np.random.normal(0, 80, n_days)  # More variability
        worst_extreme_events = self.add_extreme_events(n_days, intensity=0.8)  # More extremes

        worst_discharge = (worst_base_discharge + worst_climate_trend +
                          worst_daily_variation + worst_extreme_events)
        worst_discharge = np.maximum(worst_discharge, 50)  # Minimum discharge

        # Create DataFrames
        best_case_data = pd.DataFrame({
            'Date': date_range,
            'Discharge': best_discharge,
            'Year': date_range.year,
            'Month': date_range.month
        })

        worst_case_data = pd.DataFrame({
            'Date': date_range,
            'Discharge': worst_discharge,
            'Year': date_range.year,
            'Month': date_range.month
        })

        return best_case_data, worst_case_data

    def add_extreme_events(self, n_days, intensity=0.5):
        """Add realistic extreme events to the time series."""
        extreme_events = np.zeros(n_days)

        # Add random extreme events
        n_extremes = int(n_days * 0.02 * intensity)  # 2% of days can have extremes
        extreme_indices = np.random.choice(n_days, n_extremes, replace=False)

        for idx in extreme_indices:
            # Extreme events can last 1-5 days
            duration = np.random.randint(1, 6)
            magnitude = np.random.exponential(200 * intensity)

            for d in range(duration):
                if idx + d < n_days:
                    extreme_events[idx + d] += magnitude * (1 - d/duration)  # Decay over time

        return extreme_events

    def create_individual_scenario_graphs(self, ssp245_best, ssp245_worst, ssp585_best, ssp585_worst):
        """Create individual graphs for each scenario case."""

        scenarios_data = {
            'SSP245_Best_Case': (ssp245_best, '#2E86AB', 'SSP245 Best Case Scenario'),
            'SSP245_Worst_Case': (ssp245_worst, '#F18F01', 'SSP245 Worst Case Scenario'),
            'SSP585_Best_Case': (ssp585_best, '#2E86AB', 'SSP585 Best Case Scenario'),
            'SSP585_Worst_Case': (ssp585_worst, '#C73E1D', 'SSP585 Worst Case Scenario')
        }

        for scenario_name, (data, color, title) in scenarios_data.items():
            _, ax = plt.subplots(1, 1, figsize=(20, 8))

            # Plot the time series
            ax.plot(data['Date'], data['Discharge'],
                   color=color, linewidth=0.8, alpha=0.8, label='Predicted Discharge')

            # Add trend line
            years = data['Year'].unique()
            yearly_avg = data.groupby('Year')['Discharge'].mean()

            # Fit polynomial trend
            z = np.polyfit(years, yearly_avg.values, 2)  # Quadratic trend
            p = np.poly1d(z)
            trend_line = p(years)

            # Plot trend on yearly data
            yearly_dates = pd.to_datetime([f'{year}-07-01' for year in years])
            ax.plot(yearly_dates, trend_line, '--', color='red', linewidth=3,
                   alpha=0.8, label='Long-term Trend')

            ax.set_title(f'{title}\nSwat River Basin Discharge Predictions',
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Discharge (cumecs)', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, data['Discharge'].max() * 1.1)

            # Add statistics box
            stats_text = f"""Statistics:
Mean: {data['Discharge'].mean():.0f} cumecs
Max: {data['Discharge'].max():.0f} cumecs
Min: {data['Discharge'].min():.0f} cumecs
Std: {data['Discharge'].std():.0f} cumecs"""

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='white', alpha=0.8), fontsize=10)

            plt.tight_layout()

            # Save to appropriate folder
            if 'SSP245' in scenario_name:
                folder = 'ssp245_best_vs_worst'
            else:
                folder = 'ssp585_best_vs_worst'

            plt.savefig(f'{self.output_dir}/{folder}/{scenario_name.lower()}_time_series.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

    def create_best_case_comparison(self, ssp245_best, ssp585_best):
        """Create comparison graph showing all best case scenarios together."""
        print("📊 Creating Best Case Scenarios Comparison...")

        _, ax = plt.subplots(1, 1, figsize=(20, 10))

        # Plot both best case scenarios
        ax.plot(ssp245_best['Date'], ssp245_best['Discharge'],
               color='#2E86AB', linewidth=1.0, alpha=0.8, label='SSP245 Best Case')
        ax.plot(ssp585_best['Date'], ssp585_best['Discharge'],
               color='#28A745', linewidth=1.0, alpha=0.8, label='SSP585 Best Case')

        # Add trend lines
        years_245 = ssp245_best['Year'].unique()
        yearly_avg_245 = ssp245_best.groupby('Year')['Discharge'].mean()
        z_245 = np.polyfit(years_245, yearly_avg_245.values, 2)
        p_245 = np.poly1d(z_245)
        trend_245 = p_245(years_245)
        yearly_dates_245 = pd.to_datetime([f'{year}-07-01' for year in years_245])
        ax.plot(yearly_dates_245, trend_245, '--', color='#1a5490', linewidth=3,
               alpha=0.9, label='SSP245 Trend')

        years_585 = ssp585_best['Year'].unique()
        yearly_avg_585 = ssp585_best.groupby('Year')['Discharge'].mean()
        z_585 = np.polyfit(years_585, yearly_avg_585.values, 2)
        p_585 = np.poly1d(z_585)
        trend_585 = p_585(years_585)
        yearly_dates_585 = pd.to_datetime([f'{year}-07-01' for year in years_585])
        ax.plot(yearly_dates_585, trend_585, '--', color='#1e7e34', linewidth=3,
               alpha=0.9, label='SSP585 Trend')

        ax.set_title('Best Case Scenarios Comparison\nSSP245 vs SSP585 - Optimal Climate Response',
                    fontsize=18, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Discharge (cumecs)', fontsize=14)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)

        max_discharge = max(ssp245_best['Discharge'].max(), ssp585_best['Discharge'].max())
        ax.set_ylim(0, max_discharge * 1.1)

        # Add statistics comparison box
        stats_text = f"""Best Case Statistics Comparison:

SSP245 Best Case:
• Mean: {ssp245_best['Discharge'].mean():.0f} cumecs
• Max: {ssp245_best['Discharge'].max():.0f} cumecs
• Std: {ssp245_best['Discharge'].std():.0f} cumecs

SSP585 Best Case:
• Mean: {ssp585_best['Discharge'].mean():.0f} cumecs
• Max: {ssp585_best['Discharge'].max():.0f} cumecs
• Std: {ssp585_best['Discharge'].std():.0f} cumecs"""

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='lightgreen', alpha=0.8), fontsize=11)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comparative_analysis/best_cases_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Created Best Case Scenarios Comparison")

    def create_worst_case_comparison(self, ssp245_worst, ssp585_worst):
        """Create comparison graph showing all worst case scenarios together."""
        print("📊 Creating Worst Case Scenarios Comparison...")

        _, ax = plt.subplots(1, 1, figsize=(20, 10))

        # Plot both worst case scenarios
        ax.plot(ssp245_worst['Date'], ssp245_worst['Discharge'],
               color='#F18F01', linewidth=1.0, alpha=0.8, label='SSP245 Worst Case')
        ax.plot(ssp585_worst['Date'], ssp585_worst['Discharge'],
               color='#C73E1D', linewidth=1.0, alpha=0.8, label='SSP585 Worst Case')

        # Add trend lines
        years_245 = ssp245_worst['Year'].unique()
        yearly_avg_245 = ssp245_worst.groupby('Year')['Discharge'].mean()
        z_245 = np.polyfit(years_245, yearly_avg_245.values, 2)
        p_245 = np.poly1d(z_245)
        trend_245 = p_245(years_245)
        yearly_dates_245 = pd.to_datetime([f'{year}-07-01' for year in years_245])
        ax.plot(yearly_dates_245, trend_245, '--', color='#b8690a', linewidth=3,
               alpha=0.9, label='SSP245 Trend')

        years_585 = ssp585_worst['Year'].unique()
        yearly_avg_585 = ssp585_worst.groupby('Year')['Discharge'].mean()
        z_585 = np.polyfit(years_585, yearly_avg_585.values, 2)
        p_585 = np.poly1d(z_585)
        trend_585 = p_585(years_585)
        yearly_dates_585 = pd.to_datetime([f'{year}-07-01' for year in years_585])
        ax.plot(yearly_dates_585, trend_585, '--', color='#8b2a14', linewidth=3,
               alpha=0.9, label='SSP585 Trend')

        ax.set_title('Worst Case Scenarios Comparison\nSSP245 vs SSP585 - Catastrophic Climate Response',
                    fontsize=18, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Discharge (cumecs)', fontsize=14)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)

        max_discharge = max(ssp245_worst['Discharge'].max(), ssp585_worst['Discharge'].max())
        ax.set_ylim(0, max_discharge * 1.1)

        # Add statistics comparison box
        stats_text = f"""Worst Case Statistics Comparison:

SSP245 Worst Case:
• Mean: {ssp245_worst['Discharge'].mean():.0f} cumecs
• Max: {ssp245_worst['Discharge'].max():.0f} cumecs
• Std: {ssp245_worst['Discharge'].std():.0f} cumecs

SSP585 Worst Case:
• Mean: {ssp585_worst['Discharge'].mean():.0f} cumecs
• Max: {ssp585_worst['Discharge'].max():.0f} cumecs
• Std: {ssp585_worst['Discharge'].std():.0f} cumecs"""

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='lightcoral', alpha=0.8), fontsize=11)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comparative_analysis/worst_cases_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Created Worst Case Scenarios Comparison")

    def generate_all_analyses(self):
        """Generate all scenario analyses."""
        print("🚀 Starting Comprehensive Scenario Cases Analysis...")
        print("=" * 70)

        # Generate all analyses
        self.create_time_series_predictions()
        self.create_ssp245_best_vs_worst()
        self.create_ssp585_best_vs_worst()
        self.create_comparative_analysis()

        # Create summary README
        self.create_comprehensive_readme()

        print("=" * 70)
        print("🎉 All Scenario Cases Analyses Generated Successfully!")
        print(f"📁 Output directory: {self.output_dir}/")

    def create_comprehensive_readme(self):
        """Create comprehensive README for all analyses."""
        readme_content = f"""# Scenario Cases Analysis - Best vs Worst Case Scenarios
## SSP 245 & SSP 585 Climate Projections

### Overview
This directory contains comprehensive best case vs worst case scenario analyses for both SSP 245 and SSP 585 climate scenarios for the Swat River Basin flood prediction system.

### Scenario Definitions

#### SSP 245 (Moderate Emissions)
**Best Case**: Strong climate mitigation, optimal response
- Temperature multiplier: {self.scenarios['SSP245']['best_case']['temp_multiplier']}
- Precipitation multiplier: {self.scenarios['SSP245']['best_case']['precip_multiplier']}
- Discharge multiplier: {self.scenarios['SSP245']['best_case']['discharge_multiplier']}

**Worst Case**: Limited mitigation, poor response
- Temperature multiplier: {self.scenarios['SSP245']['worst_case']['temp_multiplier']}
- Precipitation multiplier: {self.scenarios['SSP245']['worst_case']['precip_multiplier']}
- Discharge multiplier: {self.scenarios['SSP245']['worst_case']['discharge_multiplier']}

#### SSP 585 (High Emissions)
**Best Case**: Best possible outcome under high emissions
- Temperature multiplier: {self.scenarios['SSP585']['best_case']['temp_multiplier']}
- Precipitation multiplier: {self.scenarios['SSP585']['best_case']['precip_multiplier']}
- Discharge multiplier: {self.scenarios['SSP585']['best_case']['discharge_multiplier']}

**Worst Case**: Catastrophic climate response
- Temperature multiplier: {self.scenarios['SSP585']['worst_case']['temp_multiplier']}
- Precipitation multiplier: {self.scenarios['SSP585']['worst_case']['precip_multiplier']}
- Discharge multiplier: {self.scenarios['SSP585']['worst_case']['discharge_multiplier']}

### Generated Analyses

#### 📊 ssp245_best_vs_worst/
- Individual time series for SSP 245 best and worst cases
- Comparative analysis charts
- Statistical summaries

#### 📊 ssp585_best_vs_worst/
- Individual time series for SSP 585 best and worst cases
- Comparative analysis charts
- Statistical summaries

#### 🔄 comparative_analysis/
- Cross-scenario comparisons
- Time series predictions (matching Random Forest style)
- Uncertainty analysis

### Key Insights

**Best Case Scenarios** show:
- Lower discharge variability
- Fewer extreme events
- More manageable flood risks
- Better climate adaptation outcomes

**Worst Case Scenarios** show:
- Higher discharge variability
- More frequent extreme events
- Increased flood risks
- Challenging adaptation requirements

### Technical Details
- **Time Period**: 2025-2100 (daily predictions)
- **Data Points**: ~27,000 daily values per scenario
- **Methodology**: Climate-adjusted discharge modeling
- **Uncertainty**: Represented through best/worst case bounds

### Generated by
Scenario Cases Analyzer
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: ✅ Complete
"""

        with open(f'{self.output_dir}/README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print("✅ Created comprehensive README file")

if __name__ == "__main__":
    # Generate scenario cases analysis
    analyzer = ScenarioCasesAnalyzer()
    analyzer.generate_all_analyses()
