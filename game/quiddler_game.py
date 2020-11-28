import pygtrie as trie
from collections import namedtuple

CardWord = namedtuple('CardWord' , 'score count cards')
ScoredPlay = namedtuple('ScoredPlay', 'play_score complete combo')
PlayAction = namedtuple('PlayAction', 'play pick_up drop')

card_points = {'a':2,'b':8,'c':8,'d':5,'e':2,'f':6,'g':6,'h':7,'i':2,'j':13,'k':8,
                'l':3,'m':5,'n':5,'o':2,'p':6,'q':15,'r':5,'s':3,'t':3,'u':4,'v':11,
                'w':10,'x':12,'y':4,'z':14,'qu':9,'in':7,'er':7,'cl':10,'th':9}
cards = set(card_points.keys())
double_cards = set(list(card_points.keys())[-5:])
single_cards = set(list(card_points.keys())[:-5])
double_card_first = {d[0]:d[1] for d in double_cards}

def make_perm(word, card_list, pos, collector):
    '''
    Given a word, break it into all the ways to make that word using the cards and get the
    total score for each permutation.

    For example the word: "inquiring" can be constructed from the cards in 8 ways: 
        {'in/qu/i/r/in/g': 36,
        'in/qu/i/r/i/n/g': 36,
        'in/q/u/i/r/in/g': 46,
        'in/q/u/i/r/i/n/g': 46,
        'i/n/qu/i/r/in/g': 36,
        'i/n/qu/i/r/i/n/g': 36,
        'i/n/q/u/i/r/in/g': 46,
        'i/n/q/u/i/r/i/n/g': 46}

    This is a recursive function.
        card_list holds the cards as they're identified for this word
        pos is the character position through the word
        collector is a dict where the results are stored
    '''
    while pos < len(word):
        if pos < len(word)-1:
            # check for a double letter card
            dc = word[pos]+word[pos+1]
            if dc in double_cards:
                # found a double letter card so fork here
                make_perm(word, card_list.copy()+[dc], pos+2, collector)
        card_list+=[word[pos]]
        pos+=1
    # words have to be made from 2 to 10 cards so only collect these
    if len(card_list) >= 2 and len(card_list) <= 10:
        collector['/'.join(card_list)] = sum([card_points[c] for c in card_list])

# Construct the Prefix Tree for all possible word/card permutations
with open("game/sowpods.txt", "r") as text_file:
    lines = text_file.readlines()
perms = trie.StringTrie(separator='/')
for w in lines[6:]:
    make_perm(w.strip(),[],0,perms)
print(f'Using {len(perms)} possible word permutations\n')


def find_word(word_trie, card_list, hand_left, collector):
    '''
    Find all possible words using any number and any combination of the cards
    This is a recursive function.
        word_trie is the prefix tree of all words to search over
        card_list holds the cards as they're identified for this search
        hand_left holds the cards remaining as each combination is tried
        collector is a dict where the results are stored
    '''
    for idx, _ in enumerate(hand_left):
        h = hand_left.copy()
        cards = card_list+[h.pop(idx)]
        s = '/'.join(cards)
        # if there's a prefix with these cards then carry on searching with them
        # i.e. h/e exists as a prefix but h/x doesn't so h/x would be a dead-end
        if word_trie.has_subtrie(s):
            score = word_trie.get(s)
            # if this word is complete in the trie then it has a score, so collect
            if score:
                collector[s] = CardWord(score, len(cards), cards)
            find_word(word_trie,cards,h,collector)


def scored_play(combo, hand_left):
    '''
    Score this play. Add up the total score from the word combos and subtract any score from the
    cards left in the hand. Mark as complete if all the cards have been used (empty hand).
    '''
    total_score = 0
    for word in combo:
        total_score += word[1].score
    if len(hand_left) == 0:
        return ScoredPlay(total_score, True, combo)
    else:
        return ScoredPlay(total_score-sum([card_points[c] for c in hand_left]), False, combo)   


def find_word_combos(word_trie, combo, hand_left, collector):
    '''
    Find all possible word combinations that can be made with the given hand
    This is a recursive function.
        word_trie is the prefix tree of all words to search over
        combo holds the combos as they're identified for this search
        hand_left holds the cards remaining as each combination is tried
        collector is a list where the results are stored
    '''
    # Find all possible words from the cards left in the hand
    possible_words = {}
    find_word(word_trie,[],hand_left,possible_words)
    # For each word, remove those cards from the remaining cards (hand_left) and then recurse 
    for word in possible_words.items():
        hc = hand_left.copy()
        for card in word[1].cards:
            hc.remove(card)
        cc = combo.copy()
        cc.append(word)
        find_word_combos(word_trie,cc,hc,collector)
    collector.append(scored_play(combo,hand_left))

def get_play(hand):
    '''
    Get the best play for the given hand of cards
    '''
    # Get a Prefix Tree of all words that can be made with the given hand
    possible_words = {}
    find_word(perms,[],hand,possible_words)
    pwt = trie.StringTrie(separator='/')
    pwt.update({x[0]: x[1].score for x in possible_words.items()})
    combos = []
    for w in possible_words.items():
        hc = hand.copy()
        for card in w[1].cards:
            hc.remove(card)
        find_word_combos(pwt,[w],hc,combos)
    return sorted(combos, reverse=True)[0] if len(combos) > 0 else None

def get_best_play(hand, deck_card):
    '''
    Given a hand and a deck card, return the best available play. There may not be
    a valid play, or there may not be a play that uses all the available cards.
    '''
    print(f'Hand: {"/".join(hand)}')
    print(f'Deck: {deck_card}')
    # Build up a list of all possible play options from the hand and deck card
    options = []
    # Tuple is: (Hand, Card picked up, Card dropped)
    # First option is to not pick up hence the card picked up and dropped are the same
    no_pick_up = get_play(hand)
    if no_pick_up: 
        options.append(PlayAction(no_pick_up, deck_card, deck_card))
    # Work through all cards in the hand substituting each in turn for the deck card
    for idx, _ in enumerate(hand):
        hc = hand.copy()
        hc[idx] = deck_card
        play = get_play(hc)
        if play:
            options.append(PlayAction(play, deck_card, hand[idx]))
    if options:
        best_play = sorted(options, reverse=True)[0]
        print(f'Score: {best_play.play.play_score}')
        print(f'Complete: {best_play.play.complete}')
        print(f'Words: {[c[0] for c in best_play.play.combo]}')
        print(f'Pick up: {best_play.pick_up}')
        print(f'Drop: {best_play.drop}\n')
    else:
        print('No possible play for these cards')


get_best_play(['qu','e','b','n','z'],'s')
get_best_play(['i','c','e','v','i','s','i','o','n'],'i')
get_best_play(['x','x'],'x')