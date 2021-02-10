import pygtrie as trie
from collections import namedtuple

CardWord = namedtuple('CardWord' , 'score count cards')
ScoredPlay = namedtuple('ScoredPlay', 'play_score complete combo')
PlayAction = namedtuple('PlayAction', 'play pick_up drop')

card_points = {'a':2,'b':8,'c':8,'d':5,'e':2,'f':6,'g':6,'h':7,'i':2,'j':13,'k':8,
                'l':3,'m':5,'n':5,'o':2,'p':6,'q':15,'r':5,'s':3,'t':3,'u':4,'v':11,
                'w':10,'x':12,'y':4,'z':14,'qu':9,'in':7,'er':7,'cl':10,'th':9}
double_cards = set(list(card_points.keys())[-5:])

class Quiddler(object):
    """
    Quiddler is a simple card game where you compete to make high scoring words from your hand of cards.
    Imagine Scrabble but as a card game.
    """

    def __init__(self, vocab_path="sowpods.txt"):
        # Construct the Prefix Tree for all possible word/card permutations
        with open(vocab_path, "r") as text_file:
            lines = text_file.readlines()
        self.perms = trie.StringTrie(separator='/')
        for w in lines[6:]:
            self._make_perm(w.strip(),[],0,self.perms)
        print(f'Using {len(self.perms)} possible word permutations\n')


    def _make_perm(self, word, card_list, pos, collector):
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
                    self._make_perm(word, card_list.copy()+[dc], pos+2, collector)
            card_list+=[word[pos]]
            pos+=1
        # words have to be made from 2 to 10 cards so only collect these
        if len(card_list) >= 2 and len(card_list) <= 10:
            collector['/'.join(card_list)] = sum([card_points[c] for c in card_list])

    def _find_word(self, word_trie, card_list, hand_left, collector):
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
                self._find_word(word_trie,cards,h,collector)


    def _scored_play(self, combo, hand_left):
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


    def _find_word_combos(self, word_trie, combo, hand_left, collector):
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
        self._find_word(word_trie,[],hand_left,possible_words)
        # For each word, remove those cards from the remaining cards (hand_left) and then recurse 
        for word in possible_words.items():
            hc = hand_left.copy()
            for card in word[1].cards:
                hc.remove(card)
            cc = combo.copy()
            cc.append(word)
            self._find_word_combos(word_trie,cc,hc,collector)
        collector.append(self._scored_play(combo,hand_left))

    def get_play(self, hand):
        '''
        Get the best play for the given hand of cards
        '''
        # Get a Prefix Tree of all words that can be made with the given hand
        possible_words = {}
        self._find_word(self.perms,[],hand,possible_words)
        pwt = trie.StringTrie(separator='/')
        pwt.update({x[0]: x[1].score for x in possible_words.items()})
        combos = []
        for w in possible_words.items():
            hc = hand.copy()
            for card in w[1].cards:
                hc.remove(card)
            self._find_word_combos(pwt,[w],hc,combos)
        if len(combos) > 0:
            # Custom sort key to favour fewer words
            return sorted(combos, reverse=True, key=lambda sp: sp.play_score+(5-len(sp.combo)))[0]

    def get_best_play(self, hand, deck_card):
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
        no_pick_up = self.get_play(hand)
        if no_pick_up: 
            options.append(PlayAction(no_pick_up, deck_card, deck_card))
        # Work through all cards in the hand substituting each in turn for the deck card
        for idx, _ in enumerate(hand):
            hc = hand.copy()
            hc[idx] = deck_card
            play = self.get_play(hc)
            if play:
                options.append(PlayAction(play, deck_card, hand[idx]))
        if options:
            best_play = sorted(options, reverse=True)[0]
            return {
                'score': best_play.play.play_score,
                'complete': best_play.play.complete,
                'words': [c[0] for c in best_play.play.combo],
                'pick_up': best_play.pick_up,
                'drop': best_play.drop
                }
        else:
            print('No possible play for these cards')
