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
    
    # TODO

    if voting_rule == 'plurality':
        plurality_scores = np.zeros(profiles.shape[1])
        np.add.at(plurality_scores, profiles[:, 0]-1, 1)
        winner_index = np.argmax(plurality_scores) + 1

    elif voting_rule == 'borda':
        n, m = profiles.shape       # (voters, candidates)
        borda_scores = np.zeros(m)
        individual_scores = np.arange(m-1, -1, -1)
        np.add.at(borda_scores, profiles-1, individual_scores)
        winner_index = np.argmax(borda_scores) + 1

        # Two possible alternative implementations
        # for i in range(m):
        #     np.add.at(borda_scores, profiles[:,i]-1, m-i-1)
        #     borda_scores[i] = np.sum((profiles == (i+1)) * (m - np.arange(m)))

    elif voting_rule == 'stv':
        n,m = profiles.shape
        profiles_copy = profiles.copy()
        top_candidate_indices = np.zeros(n, dtype=int)
        eliminated_candidates = np.zeros(m, dtype=int)

        while True:
    
            plurality_scores = np.zeros(m)
            # for i in range(n):
            #     plurality_scores[profiles_copy[i, top_candidate_indices[i]]-1] += 1
            np.add.at(plurality_scores, profiles_copy[np.arange(n), top_candidate_indices]-1, 1)

            if (np.max(plurality_scores) > n//2):
                winner_index = np.argmax(plurality_scores) + 1
                break
            
            plurality_scores = np.where(eliminated_candidates == 1, np.inf, plurality_scores)
            min_candidate = np.argmin(plurality_scores)
            eliminated_candidates[min_candidate] = 1
            
            for i in range(n):
                while (eliminated_candidates[profiles_copy[i, top_candidate_indices[i]] - 1] == 1):
                    top_candidate_indices[i] += 1

    elif voting_rule == 'copeland':
        n,m = profiles.shape
        copeland_scores = np.zeros(m)
        pairwise_scores = np.zeros((m,m))
        for i in range(n):
            for candidate_pair in itertools.combinations(profiles[i], 2):           
                pairwise_scores[candidate_pair[0]-1, candidate_pair[1]-1] += 1      # candidate_pair[0] has higher preference than candidate_pair[1]      
        copeland_scores = np.sum(pairwise_scores > n//2, axis=1)
        if(n%2 == 0):
            copeland_scores = copeland_scores + 0.5*np.sum(pairwise_scores == n//2, axis=1)
        winner_index = np.argmax(copeland_scores) + 1

    # END TODO

    return winner_index


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


def check_manipulable(profiles, voting_rule, find_winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        find_winner is a function that takes profiles and voting_rule as input, and gives the winner index as the output
        It is guaranteed that there will be at most 8 candidates if checking manipulability of a voting rule

        Return: Boolean representing whether the voting rule is manipulable for the given preference profiles
    '''

    manipulable = None

    # TODO
    
    manipulable = False
    n,m = profiles.shape
    winner = find_winner(profiles, voting_rule)

    if (voting_rule == 'plurality'):
        plurality_scores = np.zeros(profiles.shape[1])
        np.add.at(plurality_scores, profiles[:,0] - 1, 1)
        max_plurality_score = np.max(plurality_scores)
        for i in range(n):
            if(profiles[i, 0] == winner):
                continue
            winner_index = np.where(profiles[i] == winner)[0][0]
            for j in range(1, winner_index):
                if(plurality_scores[profiles[i,j]-1] == max_plurality_score):
                    return True
                if(plurality_scores[profiles[i,j]-1] == (max_plurality_score - 1) and profiles[i,j] < winner):
                    return True
    else:
        possible_prefs = list(itertools.permutations(np.arange(1, m+1)))
        for j in range(n):
            pref_orig = profiles[j].copy()
            for pref_new in possible_prefs:
                profiles[j] = pref_new
                new_winner = find_winner(profiles, voting_rule)
                if (np.where(pref_orig == new_winner)[0] < np.where(pref_orig == winner)[0]):
                    return True
            profiles[j] = pref_orig

    return False
    # END TODO

    return manipulable


if __name__ == '__main__':
    np.random.seed(420)

    num_tests = 200
    voting_rules = ['plurality', 'borda', 'stv', 'copeland']

    average_ranks = [[] for _ in range(len(voting_rules))]
    manipulable = [[] for _ in range(len(voting_rules))]
    for _ in tqdm(range(num_tests)):
        # Check average ranks of winner
        num_voters = np.random.choice(np.arange(80, 150))
        num_candidates = np.random.choice(np.arange(10, 80))
        profiles = generate_random_profiles(num_voters, num_candidates)

        for idx, rule in enumerate(voting_rules):
            winner = find_winner(profiles, rule)
            avg_rank = find_winner_average_rank(profiles, winner)
            average_ranks[idx].append(avg_rank / num_candidates)

        # Check if profile is manipulable or not
        num_voters = np.random.choice(np.arange(7, 11))
        num_candidates = np.random.choice(np.arange(3, 7))
        profiles = generate_random_profiles(num_voters, num_candidates)
        
        for idx, rule in enumerate(voting_rules):
            manipulable[idx].append(check_manipulable(profiles, rule, find_winner))


    # Plot average ranks as a histogram
    for idx, rule in enumerate(voting_rules):
        plt.hist(average_ranks[idx], alpha=0.8, label=rule)

    plt.legend()
    plt.xlabel('Fractional average rank of winner')
    plt.ylabel('Frequency')
    plt.savefig('average_ranks.jpg')
    
    # Plot bar chart for fraction of manipulable profiles
    manipulable = np.sum(np.array(manipulable), axis=1)
    manipulable = np.divide(manipulable, num_tests)
    plt.clf()
    plt.bar(voting_rules, manipulable)
    plt.ylabel('Manipulability fraction')
    plt.savefig('manipulable.jpg')