#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:57:59 2025

@author: davidbergman
"""


import time
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ortools.sat.python import cp_model
import numpy as np

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
        self.standardize_competencies()
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
        
        
         
    def run_model(self):
        
        model = cp_model.CpModel()
        
        persons = range(self.n_people)
        roles = range(self.n_people_per_team)
        teams = range(self.n_teams)
        
        x = {}
        for p in persons:
            for r in roles:
                for t in teams:
                    x[(p, r, t)] = model.NewBoolVar(f'x[{p},{r},{t}]')
                    if self.role_matrix[p,r] == 0:
                        model.Add(x[(p, r, t)] == 0)

        for p in persons:
            model.Add(sum(x[(p, r, t)] for r in roles for t in teams) <= 1)
            
        for r in roles:
            for t in teams:
                model.Add(sum(x[(p, r, t)] for p in persons) == 1)
          
        current_time = time.time()
        n_sols_found=0
        max_val = 0
        
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
                rand_weight = 0.0
            else:
                # print("Current time: ", current_time)
                rand_weight = (current_time - self.start_time)/self.time_limit
                # print("RAND WEIGHT: ", rand_weight)
            # print("\trandom weight: ", rand_weight)
            # print("\tmax val: ", max_val)
            # sys.exit(1)
       
            individual_efficacy = []
            for p in persons:
                for r in roles:
                    for t in teams:
                        individual_efficacy.append((self.individual_role_weights[p,r]*self.position_weights[r] + rand_weight*np.random.uniform(-1*max_val/(self.n_people_per_team*self.n_teams),max_val/(self.n_people_per_team*self.n_teams)))*x[(p, r, t)])
            
            model.Maximize(sum(individual_efficacy))
            # q = {(p, r, t): (p + 1) * (r + 1) * (t + 1) for p in persons for r in roles for t in teams}
            # model.Maximize(sum(q[(p, r, t)] * x[(p, r, t)] for p in persons for r in roles for t in teams))
        
            # Solve the model
            solver = cp_model.CpSolver()
            status = solver.Solve(model)    
            
            if n_sols_found==0:
                max_val = solver.ObjectiveValue()
            
        
            # Output the results
            if status == cp_model.OPTIMAL:
                
                # print("\tobjective value: ", solver.ObjectiveValue())
                # print("\tgetting solutions")
                this_solution = np.zeros(shape=(self.n_teams,self.n_people_per_team))
                # print('Optimal solution found!')
                for t in teams:
                    for r in roles:
                        for p in persons:
                            if solver.BooleanValue(x[(p, r, t)]):
                                this_solution[t,r] = p
                                # print("\t\t", f'Person {p} assigned to Role {r} on Team {t}')
                role_value,role_values,team_value,team_values = self.calculate_actual_value(this_solution)
                # print("ROLE VAL")
                # print(np.round(role_value,1))
                # print("ROLE VALUES ")
                # print(np.round(role_values,1))
                # print("TEAM VAL ")
                # print(np.round(team_value,1))
                # print("TEAM VALUES")
                # print(np.round(team_values,1))
                solution_value = self.role_weight*role_value + self.n_teams*self.n_competencies*self.team_weight*team_value
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
                    val = np.mean(individual_vals)
                elif self.teamAttr_index[a] == 'High Mean':
                    val = np.mean(individual_vals)
                elif self.teamAttr_index[a] == 'Low Mean':
                    val = -1*np.mean(individual_vals)
                elif self.teamAttr_index[a] == 'High Variance':
                    val = np.var(individual_vals)
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
        
        if status == cp_model.OPTIMAL:
            print("Optimal solution found!")
            for person in range(num_people):
                for team in range(num_teams):
                    if solver.BooleanValue(team_assignment[(person, team)]):
                        print(f'Person {person} is in team {team}')

if __name__ == "__main__":

    
    path_to_excel = "/Users/davidbergman/Library/CloudStorage/Dropbox/Workspace/ToolForTFP/2025/20250106_InputData_v2.xlsx"
    time_limit = 10
    n_teams = 3
    tfo = Optimizer(path_to_excel,time_limit,n_teams)
    tfo.prep_model()
    # sys.exit(1)
    for step in tfo.run_model():
        print(step)
