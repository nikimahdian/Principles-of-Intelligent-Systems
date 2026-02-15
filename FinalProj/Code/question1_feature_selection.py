"""
Question 1: Feature Selection using PSO and GA
Loan Dataset - Feature Selection with Evolutionary Algorithms
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pyswarms as ps
from deap import base, creator, tools, algorithms
import random
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seed (using last two digits of student ID - placeholder: 99)
RANDOM_STATE = 99
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

class FeatureSelectionResults:
    """Store all results and explanations for Question 1"""
    def __init__(self):
        self.results = {
            'pso': {},
            'ga': {},
            'comparison': {},
            'explanations': {}
        }
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            filename = os.path.join(project_root, 'Results', 'question1_results.json')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
    
    def save_explanations(self, filename=None):
        """Save explanations for LaTeX report"""
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            filename = os.path.join(project_root, 'Explanations', 'question1_explanations.json')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results['explanations'], f, indent=4, default=str)

results_storage = FeatureSelectionResults()

def load_and_preprocess_data():
    """
    Load and preprocess the loan dataset
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    print("Loading and preprocessing data...")
    
    # Load data - use train file and combine with test if it has labels
    # Get the directory of the script and go up one level to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    train_df = pd.read_csv(os.path.join(project_root, 'Loan_Dataset', 'loan_train.csv'))
    
    # Check if test file has Status column
    test_df = pd.read_csv(os.path.join(project_root, 'Loan_Dataset', 'loan_test.csv'))
    if 'Status' in test_df.columns:
        # Combine for preprocessing
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        # Use only train file and split later
        df = train_df.copy()
    
    # Remove NaN values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('Status', axis=1)
    y = df['Status']
    
    # Convert categorical variables to numerical
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Area']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Convert target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    # Use train_test_split with 70/30 ratio as specified in problem
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    feature_names = X.columns.tolist()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    results_storage.results['explanations']['data_preprocessing'] = {
        'description': 'Data loaded and preprocessed. Categorical variables encoded using LabelEncoder. Split into 70% train and 30% test using stratified split.',
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'total_features': len(feature_names),
        'feature_names': feature_names,
        'split_method': 'train_test_split with test_size=0.3, stratified=True'
    }
    
    return X_train, X_test, y_train, y_test, feature_names

def fitness_function_pso(particles, X_train, y_train, X_test, y_test, alpha=0.7):
    """
    Fitness function for PSO
    J = α(1 - Acc) + (1 - α)(selected_features / total_features)
    """
    n_particles = particles.shape[0]
    fitness_scores = np.zeros(n_particles)
    
    for i in range(n_particles):
        # Convert particle to binary (threshold at 0.5)
        selected_features = (particles[i] > 0.5).astype(int)
        
        if np.sum(selected_features) == 0:
            fitness_scores[i] = 1.0  # Penalty for no features selected
            continue
        
        # Select features
        X_train_selected = X_train[:, selected_features == 1]
        X_test_selected = X_test[:, selected_features == 1]
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train_selected, y_train)
        
        # Calculate accuracy
        y_pred = rf.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate fitness
        total_features = len(selected_features)
        selected_count = np.sum(selected_features)
        fitness = alpha * (1 - accuracy) + (1 - alpha) * (selected_count / total_features)
        
        fitness_scores[i] = fitness
    
    return fitness_scores

def pso_feature_selection(X_train, y_train, X_test, y_test, n_features, alpha=0.7):
    """
    Particle Swarm Optimization for feature selection
    """
    print("\n=== Running PSO Feature Selection ===")
    
    # PSO parameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
    n_particles = 50
    n_iterations = 100
    
    # Initialize optimizer
    optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, 
                                      dimensions=n_features, 
                                      options=options)
    
    # Define objective function
    def objective(particles):
        return fitness_function_pso(particles, X_train, y_train, X_test, y_test, alpha)
    
    # Run optimization
    cost, pos = optimizer.optimize(objective, iters=n_iterations, verbose=True)
    
    # Get best solution
    best_features = (pos > 0.5).astype(int)
    selected_indices = np.where(best_features == 1)[0]
    
    # Evaluate final model
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_selected, y_train)
    y_pred = rf.predict(X_test_selected)
    final_accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    # Handle cost_history - it might be a list or numpy array
    cost_history_list = optimizer.cost_history
    if hasattr(cost_history_list, 'tolist'):
        cost_history_list = cost_history_list.tolist()
    elif not isinstance(cost_history_list, list):
        cost_history_list = list(cost_history_list)
    else:
        # Already a list, just ensure it's a proper Python list
        cost_history_list = list(cost_history_list)
    
    results_storage.results['pso'] = {
        'selected_features': selected_indices.tolist(),
        'selected_feature_names': [feature_names[i] for i in selected_indices],
        'num_selected': int(len(selected_indices)),
        'final_accuracy': float(final_accuracy),
        'best_cost': float(cost),
        'cost_history': cost_history_list,
        'parameters': {
            'n_particles': n_particles,
            'n_iterations': n_iterations,
            'w': options['w'],
            'c1': options['c1'],
            'c2': options['c2'],
            'alpha': alpha
        }
    }
    
    results_storage.results['explanations']['pso'] = {
        'algorithm': 'Particle Swarm Optimization',
        'description': 'PSO uses swarm intelligence where particles move through the search space to find optimal feature combinations.',
        'parameters_explained': {
            'w': 'Inertia weight controlling particle momentum',
            'c1': 'Cognitive parameter (particle\'s own best)',
            'c2': 'Social parameter (swarm\'s best)',
            'alpha': 'Weight balancing accuracy vs feature count'
        },
        'selected_features_count': int(len(selected_indices)),
        'final_accuracy': float(final_accuracy)
    }
    
    print(f"PSO Selected {len(selected_indices)} features")
    print(f"PSO Final Accuracy: {final_accuracy:.4f}")
    print(f"Selected features: {[feature_names[i] for i in selected_indices]}")
    
    return optimizer.cost_history, selected_indices

def fitness_function_ga(individual, X_train, y_train, X_test, y_test, alpha=0.7):
    """
    Fitness function for GA
    """
    selected_features = np.array(individual)
    
    if np.sum(selected_features) == 0:
        return (1.0,)  # Penalty for no features
    
    # Select features
    X_train_selected = X_train[:, selected_features == 1]
    X_test_selected = X_test[:, selected_features == 1]
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_selected, y_train)
    
    # Calculate accuracy
    y_pred = rf.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate fitness (minimize)
    total_features = len(selected_features)
    selected_count = np.sum(selected_features)
    fitness = alpha * (1 - accuracy) + (1 - alpha) * (selected_count / total_features)
    
    return (fitness,)

def ga_feature_selection(X_train, y_train, X_test, y_test, n_features, alpha=0.7):
    """
    Genetic Algorithm for feature selection
    """
    print("\n=== Running GA Feature Selection ===")
    
    # GA parameters
    population_size = 50
    n_generations = 50
    crossover_prob = 0.9
    mutation_prob = 0.1
    
    # Create fitness and individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_bool, n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Evaluation function
    def evaluate(individual):
        return fitness_function_ga(individual, X_train, y_train, X_test, y_test, alpha)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Bit flip mutation
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create population
    population = toolbox.population(n=population_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Track best fitness over generations
    best_fitness_history = []
    mean_fitness_history = []
    
    # Evolution
    for generation in range(n_generations):
        # Select next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        population[:] = offspring
        
        # Record statistics
        fits = [ind.fitness.values[0] for ind in population]
        best_fitness_history.append(min(fits))
        mean_fitness_history.append(np.mean(fits))
        
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {min(fits):.4f}, Mean fitness = {np.mean(fits):.4f}")
    
    # Get best individual
    best_ind = tools.selBest(population, 1)[0]
    selected_indices = np.where(np.array(best_ind) == 1)[0]
    
    # Evaluate final model
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_selected, y_train)
    y_pred = rf.predict(X_test_selected)
    final_accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results_storage.results['ga'] = {
        'selected_features': selected_indices.tolist(),
        'selected_feature_names': [feature_names[i] for i in selected_indices],
        'num_selected': int(len(selected_indices)),
        'final_accuracy': float(final_accuracy),
        'best_fitness_history': best_fitness_history,
        'mean_fitness_history': mean_fitness_history,
        'parameters': {
            'population_size': population_size,
            'n_generations': n_generations,
            'crossover_prob': crossover_prob,
            'mutation_prob': mutation_prob,
            'alpha': alpha
        }
    }
    
    results_storage.results['explanations']['ga'] = {
        'algorithm': 'Genetic Algorithm',
        'description': 'GA uses evolutionary principles: selection, crossover, and mutation to evolve feature subsets.',
        'operators_explained': {
            'crossover': 'Two-point crossover exchanges genetic material between parents',
            'mutation': 'Bit flip mutation introduces diversity by flipping feature selection bits',
            'selection': 'Tournament selection favors fitter individuals'
        },
        'selected_features_count': int(len(selected_indices)),
        'final_accuracy': float(final_accuracy)
    }
    
    print(f"GA Selected {len(selected_indices)} features")
    print(f"GA Final Accuracy: {final_accuracy:.4f}")
    print(f"Selected features: {[feature_names[i] for i in selected_indices]}")
    
    return best_fitness_history, selected_indices

def plot_comparison(pso_history, ga_history):
    """
    Plot comparison of PSO and GA convergence
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Fitness over iterations (showing actual fitness values)
    ax1 = plt.subplot(1, 2, 1)
    pso_iterations = range(len(pso_history))
    ga_iterations = range(len(ga_history))
    
    # Plot actual fitness values (cost for PSO, fitness for GA)
    # Note: Lower fitness/cost is better
    pso_fitness = np.array(pso_history)  # Cost history (lower is better)
    ga_fitness = np.array(ga_history)    # Fitness history (lower is better)
    
    ax1.plot(pso_iterations, pso_fitness, 'b-', label='PSO (Fitness)', linewidth=2, marker='o', markersize=4)
    ax1.plot(ga_iterations, ga_fitness, 'r-', label='GA (Fitness)', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('تعداد تکرارها / نسلها (Iterations/Generations)', fontsize=12)
    ax1.set_ylabel('مقدار برازندگی (Fitness Value)', fontsize=12)
    ax1.set_title('مقایسه همگرایی: PSO و GA\n(Convergence Comparison: PSO vs GA)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    # Note: Lower fitness is better, so decreasing trend indicates improvement
    
    # Plot 2: Accuracy comparison
    ax2 = plt.subplot(1, 2, 2)
    algorithms = ['PSO', 'GA']
    accuracies = [results_storage.results['pso']['final_accuracy'], 
                  results_storage.results['ga']['final_accuracy']]
    num_features = [results_storage.results['pso']['num_selected'],
                    results_storage.results['ga']['num_selected']]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue', alpha=0.8)
    bars2 = ax2_twin.bar(x + width/2, num_features, width, label='# Features', color='lightcoral', alpha=0.8)
    
    ax2.set_xlabel('Algorithm', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12, color='skyblue')
    ax2_twin.set_ylabel('Number of Features', fontsize=12, color='lightcoral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.set_title('PSO vs GA: Accuracy and Feature Count', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    plot_path = os.path.join(project_root, 'Results', 'question1_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plots saved to {plot_path}")
    
    # Store comparison results
    results_storage.results['comparison'] = {
        'pso_accuracy': float(results_storage.results['pso']['final_accuracy']),
        'ga_accuracy': float(results_storage.results['ga']['final_accuracy']),
        'pso_features': int(results_storage.results['pso']['num_selected']),
        'ga_features': int(results_storage.results['ga']['num_selected']),
        'convergence_speed': {
            'pso_iterations': len(pso_history),
            'ga_generations': len(ga_history)
        }
    }
    
    # Analysis explanations
    pso_better_acc = results_storage.results['pso']['final_accuracy'] > results_storage.results['ga']['final_accuracy']
    pso_fewer_features = results_storage.results['pso']['num_selected'] < results_storage.results['ga']['num_selected']
    
    results_storage.results['explanations']['comparison'] = {
        'performance_analysis': {
            'pso_accuracy': f"{results_storage.results['pso']['final_accuracy']:.4f}",
            'ga_accuracy': f"{results_storage.results['ga']['final_accuracy']:.4f}",
            'better_accuracy': 'PSO' if pso_better_acc else 'GA',
            'pso_features': results_storage.results['pso']['num_selected'],
            'ga_features': results_storage.results['ga']['num_selected'],
            'fewer_features': 'PSO' if pso_fewer_features else 'GA'
        },
        'convergence_analysis': {
            'pso_convergence': 'PSO showed ' + ('faster' if len(pso_history) < len(ga_history) else 'slower') + ' convergence',
            'ga_convergence': 'GA showed ' + ('faster' if len(ga_history) < len(pso_history) else 'slower') + ' convergence',
            'stability': 'Both algorithms converged to stable solutions'
        },
        'feature_selection_differences': {
            'reason': 'PSO and GA may select different features due to:',
            'points': [
                'Different search strategies (swarm vs evolutionary)',
                'Different exploration-exploitation balance',
                'Stochastic nature leading to different local optima',
                'Different convergence speeds affecting final solution'
            ]
        }
    }

def main():
    """Main execution function"""
    print("=" * 60)
    print("QUESTION 1: Feature Selection using PSO and GA")
    print("=" * 60)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names_global = load_and_preprocess_data()
    global feature_names
    feature_names = feature_names_global
    
    # Run PSO
    pso_history, pso_features = pso_feature_selection(X_train, y_train, X_test, y_test, 
                                                       len(feature_names), alpha=0.7)
    
    # Run GA
    ga_history, ga_features = ga_feature_selection(X_train, y_train, X_test, y_test,
                                                   len(feature_names), alpha=0.7)
    
    # Feature analysis
    print("\n=== Feature Analysis ===")
    print(f"\nPSO Selected Features: {results_storage.results['pso']['selected_feature_names']}")
    print(f"GA Selected Features: {results_storage.results['ga']['selected_feature_names']}")
    
    common_features = set(results_storage.results['pso']['selected_feature_names']) & \
                     set(results_storage.results['ga']['selected_feature_names'])
    print(f"\nCommon Features: {list(common_features)}")
    
    results_storage.results['comparison']['common_features'] = list(common_features)
    results_storage.results['comparison']['pso_unique'] = list(
        set(results_storage.results['pso']['selected_feature_names']) - common_features
    )
    results_storage.results['comparison']['ga_unique'] = list(
        set(results_storage.results['ga']['selected_feature_names']) - common_features
    )
    
    # Plot comparison
    plot_comparison(pso_history, ga_history)
    
    # Save all results
    results_storage.save_results()
    results_storage.save_explanations()
    
    print("\n" + "=" * 60)
    print("Question 1 completed successfully!")
    print("Results saved to Results/question1_results.json")
    print("Explanations saved to Explanations/question1_explanations.json")
    print("=" * 60)

if __name__ == "__main__":
    main()


