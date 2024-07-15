import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt


''' Do not change anything in this function '''
def generate_random_profiles(num_voters, num_candidates):
    '''
        Generates a NumPy array where row i denotes the strict preference order of voter i
        The first value in row i denotes the candidate with the highest preference
        Result is a NumPy array of size (num_voters x num_candidates)
    '''
    return np.array([np.random.permutation(np.arange(1, num_candidates+1)) 
            for _ in range(num_voters)])


def find_winner(profiles, voting_rule):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        In STV, if there is a tie amongst the candidates with minimum plurality score in a round, then eliminate the candidate with the lower index
        For Copeland rule, ties among pairwise competitions lead to half a point for both candidates in their Copeland score

        Return: Index of winning candidate (1-indexed) found using the given voting rule
        If there is a tie amongst the winners, then return the winner with a lower index
    '''

    winner_index = None

    if voting_rule == 'stv':
        n,m = profiles.shape
        profiles_copy = profiles.copy()

        # # Eliminate candidates with 0 plurality score in first round itself
        # plurality_scores = np.zeros(profiles.shape[1])
        # np.add.at(plurality_scores, profiles[:, 0]-1, 1)
        # for i in range(n):
        #     profiles_copy[i,:] = profiles_copy[i, np.argwhere(plurality_scores[profiles_copy[i] - 1] > 0).flatten()]

        # while True:
        #     top_candidates, plurality_scores = np.unique(profiles_copy[:,0], return_counts=True)
        #     if (max(plurality_scores) > n//2):
        #         winner_index = top_candidates[np.argmax(plurality_scores)]
        #         break
        #     else:
        #         min_plurality_score = np.min(plurality_scores)
        #         min_candidates = np.sort(top_candidates[plurality_scores == min_plurality_score])

        # n,m = profiles.shape
        # profiles_copy = profiles.copy()
        # eliminated_candidates = []
        # while True:
        #     # Calculate the plurality scores for each candidate
        #     plurality_scores = np.zeros(m)
        #     np.add.at(plurality_scores, profiles_copy[:, 0]-1, 1)
            
        #     # Check if there is a candidate with more than half of the votes
        #     if (np.max(plurality_scores) > n//2):
        #         winner_index = np.argmax(plurality_scores) + 1
        #         break
            
        #     # Find the candidates with the minimum plurality score
        #     min_plurality_score = np.min(plurality_scores)
        #     min_candidates = np.sort(np.argwhere(plurality_scores == min_plurality_score).flatten())
            
        #     # Eliminate the candidate with the lowest index among the minimum plurality score candidates
        #     eliminated_candidate = min_candidates[0]
        #     eliminated_candidates.append(eliminated_candidate)
            
        #     # Remove the eliminated candidate from the profiles
        #     profiles_copy = np.delete(profiles_copy, eliminated_candidate, axis=1)
            
        #     # Update the indices in the profiles to reflect the removed candidate
        #     for i in range(n):
        #         profiles_copy[i] = np.where(profiles_copy[i] > eliminated_candidate, profiles_copy[i] - 1, profiles_copy[i])
        
        # # Find the winner among the remaining candidates
        # remaining_candidates = np.setdiff1d(np.arange(1, m+1), eliminated_candidates)
        # winner_index = find_winner(profiles_copy, 'plurality')

        # winner_index = remaining_candidates[winner_index - 1]

        n, m = profiles.shape
        profiles_copy = profiles.copy()
        while True:
            # Calculate the plurality scores for each candidate
            plurality_scores = np.zeros(m)
            np.add.at(plurality_scores, profiles_copy[:,0]-1, 1)
            
            # Check if there is a candidate with more than half of the votes
            if (np.max(plurality_scores) > n//2):
                winner_index = np.argmax(plurality_scores) + 1
                break
            
            # Find the candidates with the minimum plurality score
            min_plurality_score = np.min(plurality_scores)
            min_candidates = np.sort(np.argwhere(plurality_scores == min_plurality_score).flatten())
            
            # Eliminate the candidate with the lowest index among the minimum plurality score candidates
            eliminated_candidate = min_candidates[0]
            
            # Mask the eliminated candidate in the profiles
            profiles_copy = np.ma.masked_equal(profiles_copy, eliminated_candidate + 1)
            
            # Update the indices in the profiles to reflect the removed candidate
            profiles_copy -= (profiles_copy > eliminated_candidate)
            
        # Find the winner among the remaining candidates
        remaining_candidates = np.ma.compressed(profiles_copy)
        winner_index = np.argmax(np.bincount(remaining_candidates)) + 1


def find_winner_average_rank(profiles, winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        winner is the index of the winning candidate for some voting rule (1-indexed)

        Return: The average rank of the winning candidate (rank wrt a voter can be from 1 to num_candidates)
    '''

    average_rank = None

    # TODO
    average_rank = np.mean(np.argwhere(profiles == winner)[:, 1] + 1)
    # END TODO

    return average_rank

if __name__ == '__main__':
    np.random.seed(420)

    num_tests = 10
    voting_rules = ['stv']

    average_ranks = [[] for _ in range(len(voting_rules))]
    manipulable = [[] for _ in range(len(voting_rules))]
    for _ in tqdm(range(num_tests)):
        # Check average ranks of winner
        num_voters = np.random.choice(np.arange(5, 6))
        num_candidates = np.random.choice(np.arange(3, 4))
        profiles = generate_random_profiles(num_voters, num_candidates)

        for idx, rule in enumerate(voting_rules):
            winner = find_winner(profiles, rule)
            avg_rank = find_winner_average_rank(profiles, winner)
            average_ranks[idx].append(avg_rank / num_candidates)


    # Plot average ranks as a histogram
    # for idx, rule in enumerate(voting_rules):
    #     plt.hist(average_ranks[idx], alpha=0.8, label=rule)

    # plt.legend()
    # plt.xlabel('Fractional average rank of winner')
    # plt.ylabel('Frequency')
    # plt.savefig('average_ranks.jpg')
