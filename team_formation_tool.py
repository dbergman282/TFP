#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:57:59 2025

@author: davidbergman
"""

#import os
#os.environ["DYLD_LIBRARY_PATH"] = "/opt/anaconda3/envs/tfp/lib"

import time
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, NonNegativeIntegers, maximize,
    SolverFactory, Binary, ConstraintList, NonNegativeReals
)
#import matplotlib.pyplot as plt
# import

import matplotlib.pyplot as plt
import io

def create_plot(self):
    # Generate the plot
    plt.scatter(
        self.bi_solution_values[:, 0], 
        self.bi_solution_values[:, 1], 
        color='blue', 
        alpha=0.6,  # Slight fade using alpha
        label="Random Solution"
    )

    plt.scatter(
        self.role_value, 
        self.team_value, 
        color='orange', 
        s=100,  # Make it slightly larger for emphasis
        label="Optimized Solution"
    )
    
    plt.title("Full Composition Results per Solution")
    plt.xlabel("Role Value (sum)")
    plt.ylabel("Team Value (sum)")
    plt.ticklabel_format(style='scientific', axis='both', scilimits=(-1, 1))
    plt.legend()

    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf



def create_test_model():
    # Create a concrete model
    model = ConcreteModel()

    # Define decision variables (integers >= 0)
    model.x = Var(domain=NonNegativeIntegers)
    model.y = Var(domain=NonNegativeIntegers)

    # Objective: Maximize 3*x + 2*y
    model.obj = Objective(expr=3 * model.x + 2 * model.y, sense=maximize)

    # Constraint: x + 2*y <= 6
    model.con = Constraint(expr=model.x + 2 * model.y <= 6)

    return model

class Optimizer:
    
    def __init__(self, df,time_limit,n_teams):
        
        self.start_time = time.time()
        self.excel_file = df
        self.time_limit = time_limit
        self.n_teams = n_teams
        self.n_people_per_team = 5

    def prep_model(self):
        

        self.read_excel()
        self.get_n_people_and_n_competencies_and_n_teamAttr()
        # self.standardize_competencies()
        self.create_role_matrix()
        self.create_weights()


    def run_optimization(self):

        
        self.run_model()
        sys.exit(1)
        self.test_model()
        sys.exit(1)
        

        for i in range(1, 100):
            time.sleep(0.5)  # Simulate computation time
            yield f"Step {i}: Optimizing with BOO".strip()
            print("HI")

        # Return the results after completion
        return {
            "optimized_value_1": 3,
            "optimized_value_2": 4,
            "status": "Success"
        }
    
    def read_excel(self):
        self.data_dict = {}
        excel_data = pd.ExcelFile(self.excel_file)
        for sheet in excel_data.sheet_names:
            self.data_dict[sheet] = pd.read_excel(excel_data, sheet_name=sheet, index_col=0)
            
    def get_n_people_and_n_competencies_and_n_teamAttr(self):

        self.n_people = self.data_dict['Individual Comp'].shape[0]
        self.n_competencies = self.data_dict['Individual Comp'].shape[1]
        if self.n_people < self.n_people_per_team * self.n_teams:
            raise ValueError("The number of people does not exceed the number of people per team times team size")
        self.n_teamAttr = self.data_dict['Team Attributes Values'].shape[1]
        # print(self.n_teamAttr)
        # sys.exit(1)
            
    def standardize_competencies(self):
        
        self.scalers = {}

        for comp in self.data_dict['Individual Comp'].columns:
            scaler = StandardScaler()
            # print(comp)
            self.data_dict['Individual Comp'][comp + "_scaled"] = scaler.fit_transform(self.data_dict['Individual Comp'][[comp]])
            self.scalers[comp] = scaler
            
        for teamAttr in self.data_dict['Team Attributes Values'].columns:
            scaler = StandardScaler()
            # print(teamAttr)
            self.data_dict['Team Attributes Values'][teamAttr + "_scaled"] = scaler.fit_transform(self.data_dict['Team Attributes Values'][[teamAttr]])
            self.scalers[teamAttr] = scaler
            
        # for team_attribute in s
        
    # Function to sample rows
    def sample_rows(self,matrix):
        indices = []
        for col in range(matrix.shape[1]):
            # Get the indices of rows where column 'col' is 1
            valid_indices = np.where(matrix[:, col] == 1)[0]
            
            if len(valid_indices) == 0:
                raise ValueError(f"No rows with a 1 in column {col}")
            
            # Randomly select one row index from the valid indices
            selected_index = np.random.choice(valid_indices)
            indices.append(selected_index)
        
        return indices
         
    def create_role_matrix(self):
        
        self.role_matrix = np.array(self.data_dict['Individual Roles Allowed'])

        if self.role_matrix.shape[0] != self.n_people:
            raise ValueError("role_matrix does not have the right number or rows ")

        if self.role_matrix.shape[1] != self.n_people_per_team:
            raise ValueError("role_matrix does not have the right number of columns ")
        # sys.exit(1)
        
    def create_weights(self):
        
        self.position_weights =  np.array(self.data_dict['Role Imp'])[0:5,0]
        # print(self.position_weights)
        # print(self.position_weights.shape)
        if np.sum(self.position_weights) != 100:
            raise ValueError("position weights do not add to 1")
            
        self.comp_role_weights = np.array(self.data_dict['Role x Comp Matrix'])[0:self.n_competencies,1:6]
        role_weights = self.comp_role_weights.sum(axis=0)
        if np.any(role_weights != 100):
            raise ValueError("at least one role weights does not sum to 1")
            
        self.individual_comp_weights = np.array(self.data_dict['Individual Comp'])[:,-11:]
        ### self.position_weights[r]
        ### self.comp_role_weights[c,r]
        ### self.individual_comp_weights[p,c]
        self.individual_role_weights = np.matmul(self.individual_comp_weights,self.comp_role_weights)
        ### self.individual_role_weights[p,r]
        # sys.exit(1)
        # sys.exit(1)
        # sys.exit(1)
        self.individual_teamAttr_weights = np.array(self.data_dict['Team Attributes Values'])[:,-8:]
        self.teamAttr_weights = np.array(self.data_dict['Team Attributes'])[0:-1,-1]
        self.teamAttr_index = np.array(self.data_dict['Team Attributes'])[0:-1,-2]
        
        self.team_weight = self.data_dict['Team v Role Impt'].loc['Team'].values[0]
        # print(self.team_weight)
        self.role_weight = self.data_dict['Team v Role Impt'].loc['Roles'].values[0]
        # print(self.role_weight)
        # sys.exit(1)
        ### self.individual_teamAttr_weights[p,a]
        ### self.teamAttr_weights[a]
        ### self.teamAttr_index[a]
        # sys.exit(1)
        
    def generate_random_matrices(self,role_matrix, num_matrices, num_rows):
        
        num_cols = role_matrix.shape[1]
        matrices_dict = {}
    
        for i in range(num_matrices):
            random_matrix = np.zeros((num_rows, num_cols), dtype=int)
    
            for col in range(num_cols):
                # Find all indices in the big matrix where the column has a 1
                valid_indices = np.where(role_matrix[:, col] == 1)[0]
    
                if len(valid_indices) < num_rows:
                    raise ValueError(f"Not enough valid rows in column {col} for the given number of rows.")
    
                # Randomly select 'num_rows' unique indices
                selected_indices = np.random.choice(valid_indices, size=num_rows, replace=False)
                random_matrix[:, col] = selected_indices  # Assign the selected indices to the column
    
            matrices_dict[i] = random_matrix
    
        return matrices_dict

        
        
         
    def run_model(self):
        
        # model= create_test_model()
        # solver = SolverFactory("cbc")  # or 'glpk', 'gurobi', etc.

        # # Solve the model
        # results = solver.solve(model, tee=True)
        # sys.exit(1)
        
        np.random.seed(123)
        # self.role_weight = 100
        # self.team_weight = 0

        yield ""
        yield "Generating initial pool of 10000 random solutions ... "
        yield ""
        # self.n_random = 1000
        # self.rows = self.sample_rows(self.role_matrix)
        # # Perform the sampling 1000 times
        self.num_samples = 10000
        self.random_matrices = self.generate_random_matrices(self.role_matrix, self.num_samples, self.n_teams)

        self.solution_values = []
        self.bi_solution_values = []
        self.results = []
        for solution_index in range(self.num_samples):
            if solution_index % 1000 == 0:
                yield f"\tGenerated {solution_index} out of {self.num_samples}"
                # print("\t", solution_index)
            solution = self.random_matrices[solution_index]
            role_value,role_values,team_value,team_values = self.calculate_actual_value(solution)
            solution_value = self.role_weight*role_value + self.team_weight*team_value
            self.results.append([role_value,role_values,team_value,team_values])
            self.solution_values.append(solution_value)
            self.bi_solution_values.append([role_value,team_value])
        
        # sys.exit(1)
        
        self.solution_values = np.array(self.solution_values)
        self.bi_solution_values = np.array(self.bi_solution_values)
        
        self.mean_solution_value = np.mean(self.solution_values)
        self.mean_bi_solution_values = np.mean(self.bi_solution_values,axis=0)
        
        self.var_solution_value = np.var(self.solution_values)
        self.var_bi_solution_values = np.var(self.bi_solution_values,axis=0)
        
        self.calibrated_team_multiplier = self.mean_bi_solution_values[0]/self.mean_bi_solution_values[1]
        # self.calibrated_team_multiplier = self.calibrated_team_multiplier/1000
        # self.calibrated_team_multiplier = 0

        self.solution_values_calibrated = self.role_weight*self.bi_solution_values[:,0]*1 + self.team_weight*self.bi_solution_values[:,1]*self.calibrated_team_multiplier
        
        self.bi_solution_values_calibrated = self.bi_solution_values.copy()
        self.bi_solution_values_calibrated[:,1] = self.bi_solution_values_calibrated[:,1]*self.calibrated_team_multiplier
        
        yield "\nAverage metrics in random solutions: \n"
        yield f"\tMean overall value:  {np.round(self.mean_solution_value,2)}"
        yield f"\t\tRole value:  {np.round(self.mean_bi_solution_values[0],2)}"
        yield f"\t\tTeam value:  {np.round(self.mean_bi_solution_values[1],2)}\n"
        
        yield "\tMaximum individual metrics in random solutions:"
        yield f"\t\tRole value:  {np.round(np.max(self.bi_solution_values[:,0]))}"
        yield f"\t\tTeam value:  {np.round(np.max(self.bi_solution_values[:,1]))}"
        
        
        yield f"\nCalibrated team multiplier:  {self.calibrated_team_multiplier}"

        # self.best_random_index = np.argmax(self.solution_values)
        self.best_random_index_calibrated = np.argmax(self.solution_values_calibrated)
        
        
        yield f"\n\tBest solution (calibrated):  {self.solution_values_calibrated[self.best_random_index_calibrated]}"
        yield f"\t\tRole value:  {self.bi_solution_values[self.best_random_index_calibrated][0]}"
        yield f"\t\tTeam value:  {self.bi_solution_values[self.best_random_index_calibrated][1]}"
        yield ""
        
        
        # self.mean_individual_contribution_roles
        
        # self.calibrated_team_multiplier = self.mean_bi_solution_values[0]/self.mean_bi_solution_values[1]
        # sys.exit(1)
        # self.calibrated_team_multiplier = self.calibrated_team_multiplier

        # self.solution_values_calibrated = self.role_weight*self.bi_solution_values[:,0]*1 + self.team_weight*self.bi_solution_values[:,1]*self.calibrated_team_multiplier
        
        # # sys.exit(1)
        # self.best_random_index = np.argmax(self.solution_values)
        # self.best_random_index_calibrated = np.argmax(self.solution_values_calibrated)
        
        # yield "Completed intial pool"
        # # yield ""
        # # yield f"\tBest overall value (uncalibrated):  {self.solution_values[self.best_random_index]}"
        # # yield f"\t\tRole value:  {self.bi_solution_values[self.best_random_index][0]}"
        # # yield f"\t\tTeam value:  {self.bi_solution_values[self.best_random_index][1]}"
        # yield ""
        # yield f"\tBest overall value (calibrated):  {self.solution_values_calibrated[self.best_random_index_calibrated]}"
        # yield f"\t\tRole value:  {self.bi_solution_values[self.best_random_index_calibrated][0]}"
        # yield f"\t\tTeam value:  {self.bi_solution_values[self.best_random_index_calibrated][1]}"
        # yield ""
        # yield "Average metrics"
        # yield ""
        # yield f"\tMean overall value:  {self.mean_solution_value}"
        # yield f"\t\tRole value:  {self.mean_bi_solution_values[0]}"
        # yield f"\t\tTeam value:  {self.mean_bi_solution_values[1]}"
        # yield ""
        # yield "Max individual metrics"
        # yield ""
        # yield f"\t\tRole value:  {np.max(self.bi_solution_values[:,0])}"
        # yield f"\t\tTeam value:  {np.max(self.bi_solution_values[:,1])}"
        # yield ""
        
        # sys.exit(1)
        
        yield "Initializing optimization model ... \n"
        
        persons = range(self.n_people)
        roles = range(self.n_people_per_team)
        teams = range(self.n_teams)
        teamAttr = range(self.n_teamAttr)
        
        yield "\tCreating variables ... "
        # Create a concrete model
        model = ConcreteModel()
        
        model.constraint_list = ConstraintList()
        
        model.x = Var(persons,roles,teams,domain=Binary)
        
        model.ymax = Var(teamAttr,teams,domain=NonNegativeReals)
        model.ymin = Var(teamAttr,teams,domain=NonNegativeReals)
        
        model.zmax = Var(teamAttr,teams,persons,domain=Binary)
        model.zmin = Var(teamAttr,teams,persons,domain=Binary)
        

        # for a in teamAttr:
        #     if a != 0:
        #         continue
        #     if self.teamAttr_index[a] == 'Moderate Mean':
                
        #     # if self.te
        yield "\tAdding constriants ... "
        
        for p in persons:
            for r in roles:
                for t in teams:
                    if self.role_matrix[p,r] == 0:
                        model.constraint_list.add(model.x[p,r,t] == 0)


        for p in persons:
            person_expr = 0.0
            for r in roles:
                for t in teams:
                    person_expr += model.x[p,r,t]
            model.constraint_list.add(expr=person_expr <= 1)
  
        for r in roles:
            for t in teams:
                person_expr = 0.0
                for p in persons:
                    person_expr += model.x[p,r,t]
                model.constraint_list.add(expr=person_expr == 1)
                
        yield "\tAdding objectives and connections ... "

        objective_expr = 0.0
        for p in persons:
            for r in roles:
                for t in teams:
                    individual_role_contribution = self.individual_role_weights[p,r]*self.position_weights[r]
                    objective_expr += self.role_weight*individual_role_contribution*model.x[p,r,t]
                    
        for a in teamAttr:
            

            # print(self.teamAttr_index[a])
            
            if self.teamAttr_index[a] == 'Moderate Mean':

                for t in teams:
                    team_mean = 0.0
                    for p in persons:
                        for r in roles:
                            team_mean += self.individual_teamAttr_weights[p,a]*(1/self.n_people_per_team)*model.x[p,r,t]
                    model.constraint_list.add( model.ymax[a,t] >= team_mean-50 )
                    model.constraint_list.add( model.ymax[a,t] >= 50-team_mean )
                    objective_expr += (-1)*self.calibrated_team_multiplier*self.team_weight*self.teamAttr_weights[a]*model.ymax[a,t]
                
            elif self.teamAttr_index[a] == 'High Mean':
                for p in persons:
                    for r in roles:
                        for t in teams:
                            individual_role_contribution = self.teamAttr_weights[a]*self.individual_teamAttr_weights[p,a]*(1/self.n_people_per_team)
                            objective_expr += self.calibrated_team_multiplier*self.team_weight*individual_role_contribution*model.x[p,r,t]
                           
            elif self.teamAttr_index[a] == 'Low Mean':
                for p in persons:
                    for r in roles:
                        for t in teams:
                            individual_role_contribution = self.teamAttr_weights[a]*self.individual_teamAttr_weights[p,a]*(1/self.n_people_per_team)
                            objective_expr += (-1)*self.calibrated_team_multiplier*self.team_weight*individual_role_contribution*model.x[p,r,t]
                           
            
            elif self.teamAttr_index[a] == 'High Variance':
                # continue
                for t in teams:
                    
                    selected_person_max_expr = 0.0
                    selected_person_min_expr = 0.0
                    
                    connector_obj_max_expr = 0.0
                    connector_obj_min_expr = 0.0

                    for p in persons:
                        
                        selected_person_max_expr += model.zmax[a,t,p]
                        selected_person_min_expr += model.zmin[a,t,p]
                        
                        connector_obj_max_expr += self.individual_teamAttr_weights[p,a]*model.zmax[a,t,p]
                        connector_obj_min_expr += self.individual_teamAttr_weights[p,a]*model.zmin[a,t,p]
                        
                        selector_expr = 0.0
                        for r in roles:
                            selector_expr += model.x[p,r,t]   
                        model.constraint_list.add(model.zmax[a,t,p]<=selector_expr)
                        model.constraint_list.add(model.zmin[a,t,p]<=selector_expr)

                    model.constraint_list.add(selected_person_max_expr == 1)
                    model.constraint_list.add(selected_person_min_expr == 1)

                    model.constraint_list.add(model.ymax[a,t] == connector_obj_max_expr)
                    model.constraint_list.add(model.ymin[a,t] == connector_obj_min_expr)
                    
                    objective_expr += self.calibrated_team_multiplier*self.team_weight*self.teamAttr_weights[a]*(model.ymax[a,t] - model.ymin[a,t])
            
            else:
                print("DID NOT UNDERSTAND ATTRIBUTE INDEX")
                
                

        model.obj = Objective(expr=objective_expr, sense=maximize)
        #solver = SolverFactory("cbc",executable='/opt/anaconda3/pkgs/coincbc-2.10.5-h35dd71c_1/bin/cbc')  # or 'glpk', 'gurobi', etc.
        solver = SolverFactory("cbc")  # or 'glpk', 'gurobi', etc.
        solver.options['sec'] = self.time_limit
        
        yield "\tSolving model (searching for solutions) ... \n"
        # sys.exit(1)
        
        results = solver.solve(model, tee=False)
        
        # lower_bound = results.problem.lower_bound  # Lower bound of the solution
        # upper_bound = results.problem.upper_bound  # Upper bound of the solution
        # optimality_gap = abs(upper_bound - lower_bound) / abs(upper_bound) if upper_bound != 0 else float('inf')
      
        # # print(results)
        # print(f"Optimal solution found.")
        # print(f"Objective Value: {model.obj()}")
        # print(f"Lower Bound: {lower_bound}")
        # print(f"Upper Bound: {upper_bound}")
        # print(f"Optimality Gap: {optimality_gap * 100:.2f}%")
            
        
        # print("Objective value: ", model.obj())
        
        # yield "Optimized solution\n"
        
        # for t in teams:
        #     print(model.ymax[0,t].value)
        # sys.exit(1)
        
        this_solution = np.zeros(shape=(self.n_teams,self.n_people_per_team))
        # print('Optimal solution found!')
        for t in teams:
            for r in roles:
                for p in persons:
                    # if solver.BooleanValue(x[(p, r, t)]):
                    if model.x[p,r,t].value >0.5:
                        this_solution[t,r] = p
                        # yield "\t\t", f'Person {p} assigned to Role {r} on Team {t}'
                        print("\t\t", f'Person {p} assigned to Role {r} on Team {t}')
                        # print("\t\t\t",p, " ", r, " " , t," " ,self.individual_teamAttr_weights[p,5])
        # sys.exit(1)
        role_value,role_values,team_value,team_values = self.calculate_actual_value(this_solution)
        
        solution_val = self.role_weight*role_value + self.team_weight*self.calibrated_team_multiplier*team_value
        # print(solution_val)
        yield ""
        yield "Best optimized solution"
        yield f"\tSolution value: {np.round(solution_val)}"
        yield f"\tRole value:  {np.round(role_value)}"
        yield f"\tTeam value:  {np.round(team_value)}\t"
        best_solution = np.vectorize(lambda x: self.data_dict['Individual Roles Allowed'].index[x])(this_solution.astype(int))
        # self.best_solution = best_solution
        # sys.exit(1)
        # best_solution_person_index = np.array([
        #     self.data_dict['Individual Comp'].iloc[row_indices].to_numpy()
        #     for row_indices in best_solution
        # ])
        # self.best_solution_person_index = best_solution_person_index
        
        index_labels = ['Team_' + str(x+1) for x in range(self.n_teams)]
        column_labels = self.data_dict['Individual Roles Allowed'].columns
        
        self.role_value = role_value 
        self.team_value = team_value
        
        solution_df = pd.DataFrame(data=best_solution, index=index_labels, columns=column_labels)
        self.solution_df = solution_df
        return solution_df
        sys.exit(1)
        # print("Role value: ", role_value)
        # print("\trole values: ", role_values)
        # print("Team value: ", team_value)
        # print("\tTeam values: ", team_values)
        
        yield "Completed intial pool"
        # yield ""
        # yield f"\tBest overall value (uncalibrated):  {self.solution_values[self.best_random_index]}"
        # yield f"\t\tRole value:  {self.bi_solution_values[self.best_random_index][0]}"
        # yield f"\t\tTeam value:  {self.bi_solution_values[self.best_random_index][1]}"
        yield ""
        yield f"\tBest overall value (calibrated):  {self.solution_values_calibrated[self.best_random_index_calibrated]}"
        yield f"\t\tRole value:  {self.bi_solution_values[self.best_random_index_calibrated][0]}"
        yield f"\t\tTeam value:  {self.bi_solution_values[self.best_random_index_calibrated][1]}"
        yield ""
        yield "Average metrics"
        yield ""
        yield f"\tMean overall value:  {self.mean_solution_value}"
        yield f"\t\tRole value:  {self.mean_bi_solution_values[0]}"
        yield f"\t\tTeam value:  {self.mean_bi_solution_values[1]}"
        yield ""
        yield "Max individual metrics"
        yield ""
        yield f"\t\tRole value:  {np.max(self.bi_solution_values[:,0])}"
        yield f"\t\tTeam value:  {np.max(self.bi_solution_values[:,1])}"
        yield ""
        
        

        # print("here")
        sys.exit(1)




        # Define decision variables (integers >= 0)
        model.x = Var(domain=NonNegativeIntegers)
        model.y = Var(domain=NonNegativeIntegers)

        # Objective: Maximize 3*x + 2*y
        model.obj = Objective(expr=3 * model.x + 2 * model.y, sense=maximize)

        # Constraint: x + 2*y <= 6
        model.con = Constraint(expr=model.x + 2 * model.y <= 6)

        
        sys.exit(1)
        

        self.mean_individual_contribution_roles = self.mean_bi_solution_values[0]/(self.n_teams*self.n_people_per_team)
        self.mean_individual_contribution_teams = self.mean_bi_solution_values[1]/(self.n_teams*self.n_people_per_team)
        # print(self.mean_individual_contribution_roles)
        # print(self.mean_individual_contribution_teams)
        # sys.exit(1)
        # sys.exit(1)
        
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise Exception("SCIP solver is not available.")
        
        
        self.role_weight = 0
        self.team_weight = 100
        
        persons = range(self.n_people)
        roles = range(self.n_people_per_team)
        teams = range(self.n_teams)
        teamAttr = range(self.n_teamAttr)
        
        x = {}
        ymax = {}
        ymin = {}
        
        for p in persons:
            for r in roles:
                for t in teams:
                    x[(p, r, t)] = solver.IntVar(0,1,f'x[{p},{r},{t}]')
                    if self.role_matrix[p,r] == 0:
                        solver.Add(x[(p, r, t)] == 0)
        for t in teams:
            for a in teamAttr:
                ymax[(t,a)] = solver.IntVar(0,100,f'ymax[{t},{a}]')
                ymin[(t,a)] = solver.IntVar(0,100,f'ymin[{t},{a}]')

        for p in persons:
            solver.Add(sum(x[(p, r, t)] for r in roles for t in teams) <= 1)
            
        for r in roles:
            for t in teams:
                solver.Add(sum(x[(p, r, t)] for p in persons) == 1)
          
        current_time = time.time()
        n_sols_found=0
        sys.exit(1)
        
        # self.solutions = []
        self.all_solutions = []
        self.all_role_values = []
        self.all_team_values = []
        self.all_team_value = []
        self.all_role_value = []
        self.all_solution_values = []
        
        self.best_solution_index = -1
        self.best_solution_value = -1000000.0
        
        # rand_weight = 1.0
        while current_time - self.start_time < self.time_limit:
            
            if n_sols_found == 0:
                rand_multiplier = 0.0
            else:
                # print("Current time: ", current_time)
                rand_multiplier = ((current_time - self.start_time)/self.time_limit)/10
                # print("RAND WEIGHT: ", rand_weight)
            # print("\trandom weight: ", rand_weight)
            # print("\tmax val: ", max_val)
            # sys.exit(1)
       
            individual_efficacy = []
            team_efficacy = []
            
            individual_efficacy = 0
            team_efficacy = 0
            
            for p in persons:
                for r in roles:
                    for t in teams:
                        random_weight = np.random.uniform(-1*rand_multiplier,rand_multiplier)
                        individual_role_contribution = self.individual_role_weights[p,r]*self.position_weights[r]
                        objective_term = random_weight*self.role_weight*self.mean_individual_contribution_roles + individual_role_contribution

                        # print(random_weight*self.mean_individual_contribution_roles)
                        # individual_efficacy.append(objective_term*x[(p, r, t)])
                        individual_efficacy += objective_term*x[(p, r, t)]
                        solver.Objective().SetCoefficient(x[(p, r, t)], objective_term)
                        # individual_efficacy.append((self.individual_role_weights[p,r]*self.position_weights[r] + rand_weight*np.random.uniform(-1*max_val/(self.n_people_per_team*self.n_teams),max_val/(self.n_people_per_team*self.n_teams)))*x[(p, r, t)])
                      
            
            for t in teams:
                continue
                for a in teamAttr:
                    if self.teamAttr_index[a] == 'Moderate Mean': 
                        continue
                        # personal_team_weight += 0
                        mean_expression = 0
                        for p in persons:
                            mean_expression += int(self.team_weight*self.teamAttr_weights[a]*self.individual_teamAttr_weights[p,a]*(1/self.n_people_per_team))*x[(p, r, t)]
                        solver.Add(ymax[(t,a)] >= mean_expression-50)
                        solver.Add(ymax[(t,a)] >= 50-mean_expression)
                        team_efficacy.append(-1*ymax[(t,a)])
                        print(mean_expression)

                    elif self.teamAttr_index[a] == 'High Mean': 
                        # print("HIGH MEAN")
                        for p in persons:
                            team_efficacy += self.calibrated_team_multiplier*self.teamAttr_weights[a]*self.individual_teamAttr_weights[p,a]*(1/self.n_people_per_team)*x[(p, r, t)]
                            # team_efficacy.append(self.calibrated_team_multiplier*self.team_weight*self.teamAttr_weights[a]*self.individual_teamAttr_weights[p,a]*(1/self.n_people_per_team)*x[(p, r, t)])
                            # team_efficacy.append(self.calibrated_team_multiplier*self.teamAttr_weights[a]*self.individual_teamAttr_weights[p,a]*(1/self.n_people_per_team)*x[(p, r, t)])
                    elif self.teamAttr_index[a] == 'Low Mean':
                        continue
                        for p in persons:
                            # print(self.team_weight)
                            # print(self.teamAttr_weights[a])
                            # print(self.individual_teamAttr_weights[p,a])
                            # print(1/self.n_people_per_team)
                            # term = int(self.team_weight*(-1)*self.teamAttr_weights[a]*self.individual_teamAttr_weights[p,a]*(1/self.n_people_per_team))
                            # team_efficacy.append()
                            # print(int(self.team_weight*(-1)*self.teamAttr_weights[a]*self.individual_teamAttr_weights[p,a]*(1/self.n_people_per_team)))
                            team_efficacy.append(int(self.calibrated_team_multiplier*self.team_weight*(-1)*self.teamAttr_weights[a]*self.individual_teamAttr_weights[p,a]*(1/self.n_people_per_team))*x[(p, r, t)])
                    elif self.teamAttr_index[a] == 'High Variance':
                        # continue
                        continue
                        if np.random.rand() < 0.5:  
                            team_efficacy.append(int(self.calibrated_team_multiplier*self.team_weight*self.teamAttr_weights[a]*self.individual_teamAttr_weights[p,a])*x[(p, r, t)])
                        else:
                            team_efficacy.append(int(self.calibrated_team_multiplier*self.team_weight*(-1)*self.teamAttr_weights[a]*self.individual_teamAttr_weights[p,a])*x[(p, r, t)])

            
            # for p in persons:
            #     personal_team_weight = 0
            #     for a in teamAttr:  
            #         if self.teamAttr_index[a] == 'Moderate Mean': 
            #             personal_team_weight += 0
            #         elif self.teamAttr_index[a] == 'High Mean': 
            #             personal_team_weight += self.individual_teamAttr_weights[p,a]
            #         elif self.teamAttr_index[a] == 'Low Mean':
            #             personal_team_weight += -1*self.individual_teamAttr_weights[p,a]
            #         elif self.teamAttr_index[a] == 'High Variance':
            #             if np.random.rand() < 0.5:    
            #                 personal_team_weight += self.individual_teamAttr_weights[p,a]
            #             else:
            #                 personal_team_weight += -1*self.individual_teamAttr_weights[p,a]
            #         else:
            #             print("DID NOT UNDERSTAND ATTRIBUTE INDEX")
            #     for r in roles:
            #         for t in teams:  
            #             team_efficacy.append(self.calibrated_team_multiplier*personal_team_weight*x[(p, r, t)])
                
            # full_objective = individual_efficacy + team_efficacy
            # for var in full_objective:
            #     solver.Objective().SetCoefficient(var, full_objective[var])  # Set the coefficient for each variable

            # solver.Objective().SetLinearExpression(full_objective)
            
            solver.Objective().SetMaximization()
            # print(individual_efficacy)
            # sys.exit(1)
            # q = {(p, r, t): (p + 1) * (r + 1) * (t + 1) for p in persons for r in roles for t in teams}
            # model.Maximize(sum(q[(p, r, t)] * x[(p, r, t)] for p in persons for r in roles for t in teams))
        
            # Solve the model
            # solver = cp_model.CpSolver()
            # solver = pywraplp.Solver.CreateSolver('SCIP')
            # Create the SCIP solver
            
            
            solver = pywraplp.Solver.CreateSolver('SCIP')
            if not solver:
                raise Exception("SCIP solver is not available.")
            status = solver.Solve()    
            print(status)
            
            # print(solver.ObjectiveValue())
            # print(solver.Objective().Value())
            # sys.exit(1)
            # Output the results
            if status == pywraplp.Solver.OPTIMAL:
                
                # print("\tobjective value: ", solver.ObjectiveValue())
                # print("\tgetting solutions")
                this_solution = np.zeros(shape=(self.n_teams,self.n_people_per_team))
                # print('Optimal solution found!')
                for t in teams:
                    for r in roles:
                        for p in persons:
                            # if solver.BooleanValue(x[(p, r, t)]):
                            if x[(p, r, t)].solution_value()>0.5:
                                this_solution[t,r] = p
                                print("\t\t", f'Person {p} assigned to Role {r} on Team {t}')
                sys.exit(1)
                role_value,role_values,team_value,team_values = self.calculate_actual_value(this_solution)

                # solution_value = self.role_weight*role_value + self.n_teams*self.n_competencies*self.team_weight*team_value
                print(role_value)
                print(team_value)
                solution_value = self.role_weight*role_value + self.calibrated_team_multiplier*team_value
                # print(solution_val)
                self.all_solutions.append(this_solution)
                self.all_role_values.append(role_values)
                self.all_team_values.append(team_values)
                self.all_team_value.append(team_value)
                self.all_role_value.append(role_value)
                self.all_solution_values.append(solution_value)
                # found_better_solution = False
                if solution_value > self.best_solution_value:
                    self.best_solution_value = solution_value
                    self.best_solution_index = n_sols_found
                    # found_better_solution = True
                n_sols_found+=1
                current_time = time.time()
                time_elapsed = np.round(current_time - self.start_time,1)
                best_solution_value = np.round(self.all_solution_values[self.best_solution_index])
                best_solution_role_value = np.round(self.all_role_value[self.best_solution_index])
                best_solution_team_value = np.round(self.all_team_value[self.best_solution_index])
                
                yield f"Completed candidate {n_sols_found} in {time_elapsed} seconds"
                yield f"\tBest solution index: {self.best_solution_index}"
                yield f"\tBest value: {best_solution_value} (role: {best_solution_role_value}, team: {best_solution_team_value})"
                # sys.exit(1)
                # print("\tsolution got")
                # sys.exit(1)
            else:
                raise ValueError("No possible solution")
                # print('No Possible Solution.')
        best_solution = self.all_solutions[self.best_solution_index]
        best_solution = np.vectorize(lambda x: self.data_dict['Individual Roles Allowed'].index[x])(best_solution.astype(int))
        # self.best_solution = best_solution
        # sys.exit(1)
        # best_solution_person_index = np.array([
        #     self.data_dict['Individual Comp'].iloc[row_indices].to_numpy()
        #     for row_indices in best_solution
        # ])
        # self.best_solution_person_index = best_solution_person_index
        
        index_labels = ['Team_' + str(x+1) for x in range(self.n_teams)]
        column_labels = self.data_dict['Individual Roles Allowed'].columns
        
        
        
        solution_df = pd.DataFrame(data=best_solution, index=index_labels, columns=column_labels)
        self.solution_df = solution_df
        return solution_df
                
    def calculate_actual_value(self,solution):
        # print(solution)

        # persons = range(self.n_people)
        roles = range(self.n_people_per_team)
        teams = range(self.n_teams)
        teamAttrs = range(self.n_teamAttr)
        
        role_values = np.zeros(self.n_teams)
        for t in teams:
            for r in roles:

                p = int(solution[t,r])
                role_values[t] += self.individual_role_weights[p,r]*self.position_weights[r]
        
        team_values = np.zeros(self.n_teams)
        for t in teams:
            for a in teamAttrs:

                # print(self.teamAttr_index[a])
                # print("team: ", t," attr: ", a)
                # print(self.teamAttr_weights[a])
                # print(self.teamAttr_index[a])
                individual_vals = np.zeros(self.n_people_per_team)
                for r in roles:
                    p = int(solution[t,r])
                    # print(self.individual_teamAttr_weights[p,a])
                    # print(a)
                    # print(p)
                    # print(self.individual_teamAttr_weights[p,a])
                    # print(individual_vals[r])
                    individual_vals[r] = self.individual_teamAttr_weights[p,a]
                if self.teamAttr_index[a] == 'Moderate Mean':
                    # val = np.mean(individual_vals)
                    val = -1*np.abs(50-np.mean(individual_vals))
                elif self.teamAttr_index[a] == 'High Mean':
                    val = np.mean(individual_vals)
                elif self.teamAttr_index[a] == 'Low Mean':
                    val = -1*np.mean(individual_vals)
                elif self.teamAttr_index[a] == 'High Variance':
                    # continue
                    val = np.max(individual_vals) - np.min(individual_vals)
                    # val = np.var(individual_vals)
                else:
                    print("DID NOT UNDERSTAND ATTRIBUTE INDEX")
                # print(self.teamAttr_weights[a])
                team_values[t] += self.teamAttr_weights[a]*val

        # print("end")
        # print(collective_team_values)    
        # sys.exit(1)
        
        
        return np.sum(role_values),role_values, np.sum(team_values), team_values
        # sys.exit(1)
        
    def test_model(self):
        model = cp_model.CpModel()

        # Example: 6 people, 3 teams
        num_people = 6
        num_teams = 3
        
        # Decision variables: person i is assigned to team j (0/1 variable)
        team_assignment = {}
        for person in range(num_people):
            for team in range(num_teams):
                team_assignment[(person, team)] = model.NewBoolVar(f'person_{person}_team_{team}')
        
        # Constraint: Each person is assigned to exactly one team
        for person in range(num_people):
            model.Add(sum(team_assignment[(person, team)] for team in range(num_teams)) == 1)
        
        # Constraint: Team sizes (optional, e.g., max 3 per team)
        for team in range(num_teams):
            model.Add(sum(team_assignment[(person, team)] for person in range(num_people)) <= 3)
        
        # Objective: Maximize some team quality (example: random weights)
        team_quality = {team: model.NewIntVar(0, 100, f'team_quality_{team}') for team in range(num_teams)}
        for team in range(num_teams):
            team_quality[team] = sum(team_assignment[(person, team)] * (person + 1) for person in range(num_people))  # Example
        
        # Aggregate team quality (e.g., maximize the sum of all team qualities)
        model.Maximize(sum(team_quality[team] for team in range(num_teams)))
        
        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # print("SOLUTION VALUE: ", model.objVal)
        
        if status == cp_model.OPTIMAL:
            print("Optimal solution found!")
            for person in range(num_people):
                for team in range(num_teams):
                    if solver.BooleanValue(team_assignment[(person, team)]):
                        print(f'Person {person} is in team {team}')

if __name__ == "__main__":

    
    # path_to_excel = "/Users/davidbergman/Library/CloudStorage/Dropbox/Workspace/ToolForTFP/2025/20250114_InputData_v2_JM Real.xlsx"
    # path_to_excel = "/Userx`s/davidbergman/Library/CloudStorage/Dropbox/Workspace/ToolForTFP/2025/20250116_InputData_v2_JM Multiple Positions.xlsx"
    path_to_excel = "/Users/davidbergman/Library/CloudStorage/Dropbox/Workspace/ToolForTFP/2025/20250116_InputData_v2_JM Solo Positions.xlsx"
    time_limit = 60
    n_teams = 10
    tfo = Optimizer(path_to_excel,time_limit,n_teams)
    tfo.prep_model()
    # sys.exit(1)
    for step in tfo.run_model():
        print(step)
    
