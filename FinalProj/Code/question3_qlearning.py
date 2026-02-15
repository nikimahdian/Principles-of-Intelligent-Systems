"""
Question 3: Q-Learning Theory and Analysis
Reinforcement Learning - Q-Learning Concepts
"""

import json
import os
import numpy as np

class QLearningResults:
    """Store all results and explanations for Question 3"""
    def __init__(self):
        self.results = {
            'part_a': {},
            'part_b': {},
            'part_c': {},
            'explanations': {}
        }
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            filename = os.path.join(project_root, 'Results', 'question3_results.json')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
    
    def save_explanations(self, filename=None):
        """Save explanations for LaTeX report"""
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            filename = os.path.join(project_root, 'Explanations', 'question3_explanations.json')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results['explanations'], f, indent=4, default=str)

results_storage = QLearningResults()

def part_a_qlearning_equation():
    """
    Part A: Q-learning equation and parameters
    """
    print("\n=== Part A: Q-Learning Equation and Parameters ===")
    
    # Q-learning update equation
    q_equation = r"Q_{t+1}(s_t, a_t) = (1 - \alpha) Q_t(s_t, a_t) + \alpha(r_t + \gamma \max_{a'} Q_t(s_{t+1}, a'))"
    
    # Parameter explanations
    parameters = {
        'alpha': {
            'symbol': r'$\alpha$',
            'name': 'Learning Rate',
            'description': 'Controls how much new information overrides old information. Range: [0, 1]',
            'role': 'Determines the step size of Q-value updates. Higher α means faster learning but potentially less stable. Lower α means slower but more stable convergence.',
            'typical_values': 'Common values: 0.1 to 0.5. Often decayed over time (e.g., α_t = α_0 / (1 + t))'
        },
        'gamma': {
            'symbol': r'$\gamma$',
            'name': 'Discount Factor',
            'description': 'Determines the importance of future rewards. Range: [0, 1]',
            'role': 'Higher γ (close to 1) makes the agent value long-term rewards more. Lower γ (close to 0) makes the agent focus on immediate rewards. Essential for handling infinite-horizon problems.',
            'typical_values': 'Common values: 0.9 to 0.99. γ=0 means only immediate reward matters, γ=1 means future rewards are equally important.'
        },
        'epsilon': {
            'symbol': r'$\epsilon$',
            'name': 'Epsilon (ε-greedy policy)',
            'description': 'Probability of selecting a random action instead of the greedy action. Range: [0, 1]',
            'role': 'Balances exploration vs exploitation. ε=1 means pure exploration (random), ε=0 means pure exploitation (greedy). Often decayed over time to shift from exploration to exploitation.',
            'typical_values': 'Common values: Start with 0.9-1.0, decay to 0.01-0.1. Decay schedule: ε_t = max(ε_min, ε_max * decay^t)'
        }
    }
    
    # Off-policy explanation
    off_policy_explanation = {
        'concept': 'Q-learning is an off-policy algorithm',
        'definition': 'Off-policy means the algorithm learns the value of the optimal policy while following a different (behavior) policy.',
        'explanation': {
            'behavior_policy': 'The policy used to select actions (e.g., ε-greedy policy)',
            'target_policy': 'The policy being learned (greedy policy: always select action with highest Q-value)',
            'key_insight': 'Q-learning updates Q-values using the maximum Q-value of the next state (greedy action), regardless of which action was actually taken. This allows learning the optimal policy while exploring.'
        },
        'contrast_with_on_policy': {
            'on_policy_example': 'SARSA is on-policy - it learns the value of the policy it follows',
            'difference': 'SARSA uses Q(s_{t+1}, a_{t+1}) where a_{t+1} is the action actually taken. Q-learning uses max_a Q(s_{t+1}, a) regardless of action taken.'
        },
        'advantages': [
            'Can learn optimal policy while exploring',
            'More sample-efficient in many cases',
            'Can use data from any policy (e.g., expert demonstrations)'
        ]
    }
    
    results_storage.results['part_a'] = {
        'q_equation': q_equation,
        'parameters': parameters,
        'off_policy_explanation': off_policy_explanation
    }
    
    results_storage.results['explanations']['part_a'] = {
        'section': 'Q-Learning Equation and Parameters',
        'equation': {
            'latex': q_equation,
            'description': 'The Q-learning update equation combines the current Q-value estimate with the new information (reward + discounted future value).'
        },
        'parameters_detailed': parameters,
        'off_policy': off_policy_explanation,
        'mathematical_interpretation': {
            'term1': {'formula': r'(1 - \alpha) Q_t(s_t, a_t)', 'description': 'Retains a fraction of the current Q-value estimate'},
            'term2': {'formula': r'\alpha(r_t + \gamma \max_{a\'} Q_t(s_{t+1}, a\'))', 'description': 'Updates with new information: immediate reward plus discounted optimal future value'},
            'balance': 'The learning rate α balances between old and new information'
        }
    }
    
    print(f"Q-Learning Equation: {q_equation}")
    print("\nParameters:")
    for param_name, param_info in parameters.items():
        print(f"\n{param_info['symbol']} ({param_info['name']}):")
        print(f"  Description: {param_info['description']}")
        print(f"  Role: {param_info['role']}")
    
    print(f"\n\nOff-Policy Explanation:")
    print(f"  {off_policy_explanation['concept']}")
    print(f"  {off_policy_explanation['definition']}")
    print(f"  Key Insight: {off_policy_explanation['explanation']['key_insight']}")

def part_b_qvalue_calculation():
    """
    Part B: Q-value calculation example
    Given: Q-table initially all zeros
    Transition: (s_t = s_0, a_t = a_1, r_t = +2, s_{t+1} = s_1)
    At same moment: max_{a'} Q(s_1, a') = 1.5
    Parameters: α = 0.2, γ = 0.9
    Calculate: Q(s_0, a_1)
    """
    print("\n=== Part B: Q-Value Calculation Example ===")
    
    # Given values
    initial_q = 0.0  # Q-table initially all zeros
    state = 's_0'
    action = 'a_1'
    reward = 2.0
    next_state = 's_1'
    max_next_q = 1.5
    alpha = 0.2
    gamma = 0.9
    
    # Q-learning update equation
    # Q_{t+1}(s_t, a_t) = (1 - α) Q_t(s_t, a_t) + α(r_t + γ max_{a'} Q_t(s_{t+1}, a'))
    
    # Step-by-step calculation
    step1_current_q = initial_q
    step2_reward_term = reward
    step3_future_value = gamma * max_next_q
    step4_td_target = step2_reward_term + step3_future_value
    step5_weighted_current = (1 - alpha) * step1_current_q
    step6_weighted_new = alpha * step4_td_target
    step7_new_q = step5_weighted_current + step6_weighted_new
    
    calculation_steps = {
        'given': {
            'initial_q_value': f'Q(s_0, a_1) = {initial_q}',
            'transition': f'(s_t = s_0, a_t = a_1, r_t = +{reward}, s_{{t+1}} = s_1)',
            'max_next_q': f'max_{{a\'}} Q(s_1, a\') = {max_next_q}',
            'alpha': f'α = {alpha}',
            'gamma': f'γ = {gamma}'
        },
        'calculation': {
            'step1': {
                'description': 'Current Q-value',
                'value': f'Q_t(s_0, a_1) = {step1_current_q}',
                'explanation': 'Initial Q-value from the Q-table (all zeros initially)'
            },
            'step2': {
                'description': 'Immediate reward',
                'value': f'r_t = {step2_reward_term}',
                'explanation': 'Reward received for taking action a_1 in state s_0'
            },
            'step3': {
                'description': 'Discounted future value',
                'value': f'γ × max_{{a\'}} Q(s_1, a\') = {gamma} × {max_next_q} = {step3_future_value}',
                'explanation': 'Discounted value of the best action in the next state'
            },
            'step4': {
                'description': 'TD Target (Temporal Difference target)',
                'value': f'r_t + γ max_{{a\'}} Q(s_1, a\') = {step2_reward_term} + {step3_future_value} = {step4_td_target}',
                'explanation': 'The target value we want to move towards'
            },
            'step5': {
                'description': 'Weighted current Q-value',
                'value': f'(1 - α) × Q_t(s_0, a_1) = (1 - {alpha}) × {step1_current_q} = {step5_weighted_current}',
                'explanation': 'Retained portion of current estimate'
            },
            'step6': {
                'description': 'Weighted new information',
                'value': f'α × TD_target = {alpha} × {step4_td_target} = {step6_weighted_new}',
                'explanation': 'New information weighted by learning rate'
            },
            'step7': {
                'description': 'New Q-value',
                'value': f'Q_{{t+1}}(s_0, a_1) = {step5_weighted_current} + {step6_weighted_new} = {step7_new_q}',
                'explanation': 'Final updated Q-value'
            }
        },
        'final_answer': {
            'value': float(step7_new_q),
            'formatted': f'Q_{{t+1}}(s_0, a_1) = {step7_new_q:.4f}'
        }
    }
    
    results_storage.results['part_b'] = {
        'given_values': calculation_steps['given'],
        'calculation_steps': calculation_steps['calculation'],
        'final_q_value': calculation_steps['final_answer']['value']
    }
    
    results_storage.results['explanations']['part_b'] = {
        'section': 'Q-Value Calculation Example',
        'problem_statement': 'Calculate the new Q-value after observing a transition, given initial Q-table (all zeros), transition details, and Q-learning parameters.',
        'step_by_step': calculation_steps,
        'interpretation': {
            'result': f'The Q-value increased from {initial_q} to {step7_new_q:.4f}, indicating that action a_1 in state s_0 leads to positive expected future rewards.',
            'learning_rate_effect': f'With α={alpha}, the update moved {alpha*100}% towards the TD target, showing moderate learning speed.',
            'discount_factor_effect': f'With γ={gamma}, future rewards are valued at {gamma*100}% of their immediate value, showing strong consideration of long-term consequences.'
        },
        'formula_application': {
            'equation': r"Q_{t+1}(s_t, a_t) = (1 - \alpha) Q_t(s_t, a_t) + \alpha(r_t + \gamma \max_{a'} Q_t(s_{t+1}, a'))",
            'substitution': f'Q_{{t+1}}(s_0, a_1) = (1 - {alpha}) × {initial_q} + {alpha} × ({reward} + {gamma} × {max_next_q})',
            'simplification': f'Q_{{t+1}}(s_0, a_1) = {step5_weighted_current} + {step6_weighted_new} = {step7_new_q:.4f}'
        }
    }
    
    print("Given:")
    for key, value in calculation_steps['given'].items():
        print(f"  {key}: {value}")
    
    print("\nCalculation Steps:")
    for step_key, step_info in calculation_steps['calculation'].items():
        print(f"\n{step_info['description']}:")
        print(f"  {step_info['value']}")
        print(f"  Explanation: {step_info['explanation']}")
    
    print(f"\n\nFinal Answer: {calculation_steps['final_answer']['formatted']}")

def part_c_convergence_issues():
    """
    Part C: Convergence issues and solutions
    """
    print("\n=== Part C: Convergence Issues and Solutions ===")
    
    convergence_issues = {
        'issue1': {
            'factor': 'Inappropriate Learning Rate (α)',
            'description': 'Learning rate that is too high or too low can cause convergence problems',
            'problems': {
                'too_high': {
                    'symptom': 'Q-values oscillate or diverge, unstable learning',
                    'cause': 'Large updates cause overshooting, Q-values bounce around optimal values',
                    'example': 'α = 1.0 causes complete replacement of old values, no learning stability'
                },
                'too_low': {
                    'symptom': 'Extremely slow convergence, requires many episodes',
                    'cause': 'Tiny updates mean Q-values change very slowly',
                    'example': 'α = 0.001 requires millions of steps to converge'
                }
            },
            'solutions': [
                {
                    'solution': 'Adaptive Learning Rate Schedule',
                    'description': 'Start with higher α (e.g., 0.5-1.0) and decay over time (e.g., α_t = α_0 / (1 + decay_rate × t))',
                    'rationale': 'Allows fast initial learning, then stabilizes as Q-values approach optimal'
                },
                {
                    'solution': 'Learning Rate Annealing',
                    'description': 'Use α_t = α_min + (α_max - α_min) × exp(-decay × t)',
                    'rationale': 'Smooth transition from exploration to exploitation'
                },
                {
                    'solution': 'State-Action Visit Count Based Learning Rate',
                    'description': 'Use α(s,a) = 1 / (1 + N(s,a)) where N(s,a) is visit count',
                    'rationale': 'Decreases learning rate for frequently visited state-action pairs, stabilizing their Q-values'
                }
            ]
        },
        'issue2': {
            'factor': 'Insufficient Exploration or Poor Exploration Strategy',
            'description': 'Inadequate exploration leads to suboptimal policies and slow convergence',
            'problems': {
                'insufficient_exploration': {
                    'symptom': 'Agent gets stuck in local optima, converges to suboptimal policy',
                    'cause': 'Early convergence to first reasonable policy found, missing better alternatives',
                    'example': 'ε-greedy with ε=0.01 from start leads to exploitation of poor initial policy'
                },
                'poor_exploration': {
                    'symptom': 'Wasteful random exploration, slow learning',
                    'cause': 'Pure random exploration doesn\'t guide learning efficiently',
                    'example': 'ε=1.0 (pure random) never exploits learned knowledge'
                }
            },
            'solutions': [
                {
                    'solution': 'Epsilon Decay Schedule',
                    'description': 'Start with high ε (0.9-1.0) and decay to low ε (0.01-0.1): ε_t = max(ε_min, ε_max × decay^t)',
                    'rationale': 'Balances exploration (early) and exploitation (later) phases'
                },
                {
                    'solution': 'Upper Confidence Bound (UCB) or Optimistic Initialization',
                    'description': 'Initialize Q-values optimistically high, encouraging exploration of unexplored states',
                    'rationale': 'Agent naturally explores less-visited states due to optimistic estimates'
                },
                {
                    'solution': 'Boltzmann (Softmax) Exploration',
                    'description': r'Use probability distribution: P(a|s) = exp(Q(s,a)/τ) / Σ exp(Q(s,a\')/τ)',
                    'rationale': 'More systematic exploration based on Q-value estimates, temperature τ controls exploration'
                }
            ]
        },
        'issue3': {
            'factor': 'Reward Scaling and Design Issues',
            'description': 'Poorly scaled or designed rewards cause learning instability',
            'problems': {
                'reward_scale': {
                    'symptom': 'Q-values explode or vanish, numerical instability',
                    'cause': 'Very large or very small rewards cause Q-values to grow/shrink unbounded',
                    'example': 'Rewards in range [0, 10000] cause Q-values to become very large'
                },
                'reward_sparsity': {
                    'symptom': 'Slow learning, especially in sparse reward environments',
                    'cause': 'Rare positive rewards provide little learning signal',
                    'example': 'Only terminal states give rewards, intermediate steps provide no feedback'
                }
            },
            'solutions': [
                {
                    'solution': 'Reward Scaling/Normalization',
                    'description': 'Normalize rewards to reasonable range (e.g., [-1, 1] or [0, 1])',
                    'rationale': 'Prevents Q-value explosion, stabilizes learning dynamics'
                },
                {
                    'solution': 'Reward Clipping',
                    'description': 'Clip rewards to fixed range: r_clipped = clip(r, r_min, r_max)',
                    'rationale': 'Prevents extreme rewards from destabilizing Q-values'
                },
                {
                    'solution': 'Reward Shaping',
                    'description': 'Design intermediate rewards to guide learning (e.g., distance-based rewards)',
                    'rationale': 'Provides learning signal throughout episode, not just at terminal states'
                }
            ]
        },
        'issue4': {
            'factor': 'State Space Discretization and Representation',
            'description': 'Poor state representation leads to slow or failed convergence',
            'problems': {
                'curse_of_dimensionality': {
                    'symptom': 'Exponentially many states, impossible to visit all',
                    'cause': 'High-dimensional or continuous state spaces create huge state spaces',
                    'example': 'Continuous states discretized too finely create millions of states'
                },
                'irrelevant_features': {
                    'symptom': 'Slow learning, poor generalization',
                    'cause': 'State representation includes irrelevant information',
                    'example': 'Including time-of-day in state when it doesn\'t affect dynamics'
                }
            },
            'solutions': [
                {
                    'solution': 'Appropriate State Discretization',
                    'description': 'Use domain knowledge to discretize continuous states meaningfully (e.g., tile coding, function approximation)',
                    'rationale': 'Reduces state space size while preserving important information'
                },
                {
                    'solution': 'Feature Selection and Engineering',
                    'description': 'Select only relevant features, use dimensionality reduction (PCA, autoencoders)',
                    'rationale': 'Reduces state space complexity, focuses learning on important aspects'
                },
                {
                    'solution': 'Function Approximation (Deep Q-Networks)',
                    'description': 'Use neural networks to approximate Q-function instead of Q-table',
                    'rationale': 'Handles high-dimensional/continuous states, enables generalization'
                }
            ]
        }
    }
    
    # Select two main issues for detailed explanation
    selected_issues = {
        'issue1': convergence_issues['issue1'],
        'issue2': convergence_issues['issue2']
    }
    
    results_storage.results['part_c'] = {
        'convergence_issues': selected_issues,
        'all_issues_summary': {k: {'factor': v['factor'], 'description': v['description']} 
                              for k, v in convergence_issues.items()}
    }
    
    results_storage.results['explanations']['part_c'] = {
        'section': 'Convergence Issues and Solutions',
        'introduction': 'Q-learning convergence can be affected by multiple factors. Two critical issues are discussed with practical solutions.',
        'detailed_issues': selected_issues,
        'summary_table': {
            'factors': [
                'Inappropriate Learning Rate',
                'Insufficient/Poor Exploration',
                'Reward Scaling Issues',
                'State Space Representation'
            ],
            'key_solutions': [
                'Adaptive learning rate schedules, visit-count based α',
                'Epsilon decay, UCB, Boltzmann exploration',
                'Reward normalization/clipping, reward shaping',
                'Appropriate discretization, feature selection, function approximation'
            ]
        },
        'general_principles': {
            'principle1': 'Balance exploration and exploitation throughout learning',
            'principle2': 'Ensure numerical stability through proper scaling',
            'principle3': 'Design state representation to capture essential information efficiently',
            'principle4': 'Use adaptive parameters that change as learning progresses'
        }
    }
    
    print("Two Key Factors Affecting Q-Learning Convergence:\n")
    
    for issue_key, issue_info in selected_issues.items():
        print(f"\n{issue_info['factor']}:")
        print(f"  Description: {issue_info['description']}")
        print("\n  Problems:")
        for prob_key, prob_info in issue_info['problems'].items():
            print(f"    - {prob_info['symptom']}")
            print(f"      Cause: {prob_info['cause']}")
        
        print("\n  Solutions:")
        for i, solution in enumerate(issue_info['solutions'], 1):
            print(f"    {i}. {solution['solution']}")
            print(f"       Description: {solution['description']}")
            print(f"       Rationale: {solution['rationale']}")

def main():
    """Main execution function"""
    print("=" * 60)
    print("QUESTION 3: Q-Learning Theory and Analysis")
    print("=" * 60)
    
    # Part A: Q-learning equation and parameters
    part_a_qlearning_equation()
    
    # Part B: Q-value calculation
    part_b_qvalue_calculation()
    
    # Part C: Convergence issues and solutions
    part_c_convergence_issues()
    
    # Save all results
    results_storage.save_results()
    results_storage.save_explanations()
    
    print("\n" + "=" * 60)
    print("Question 3 completed successfully!")
    print("Results saved to Results/question3_results.json")
    print("Explanations saved to Explanations/question3_explanations.json")
    print("=" * 60)

if __name__ == "__main__":
    main()

