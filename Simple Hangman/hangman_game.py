"""
A game of Handman.  Loads in a list of words from a text file and randomly chooses them for a game.
Pretty dirty code...a lot of work arounds.  But it works! :D
"""

import random as rand

strikes = 0
guessWord = []
guesses = []
count = 0
word = ''
stringedWord = ''

def readWords(): #reads in the textfile
    with open('hangman_words.txt', 'r') as words:
        wordList = words.readlines()
    return wordList

def pickWord(list): #picks random word
    ind = rand.randint(0,len(list)-1)
    return list[ind]

def replaceLetters(work, guessWord, guess): #recursively replaces letters because sometimes letters repeat
    if guess in work:
        ind = work.index(guess)
        guessWord[ind] = guess
        work = list(work)
        work[ind] = ''
        ''.join(work)
        replaceLetters(work, guessWord, guess)
        return work, guessWord

def generateUnder(guessWord, word, guess, count, strike, guesses): #if beginning of game, generates a list of _, otherwise fills in the _
    if count != 0:
        multipleCheck = list(guess)
        if len(multipleCheck) > 1:
            print('Only type in one letter please.\n')
        elif guess == '' or guess == " ":
            print("You didn't guess anything...\n")
        elif guess in word:
            print('Correct letter!\n')
            replaceLetters(word, guessWord, guess)
            guesses.append(guess)
        else:
            print('Wrong letter!\n')
            strike += 1
            guesses.append(guess)
        return guessWord, strike, guesses
    else:
        guessWord = ["_"] * (len(word)-1)
        return guessWord

def play(count, guessWord, strikes, word, guesses): #plays the game
    if count == 0:
        wordList = readWords()
        word = pickWord(wordList)
        print('DEBUGGING:: WORD IS:', word)
        guessWord = generateUnder(guessWord, word, 0, count, strikes, guesses)
        print('Welcome to Hangman!  Guess a letter that fills in the blank.  Six strikes and you lose!')
        print('Here is the current word:')
        print(guessWord)
        print('You have', strikes, 'strikes.')
        count = 1
        return count, guessWord, strikes, word, guesses
    else:
        print(guessWord)
        letter = input('What is your guess? ')
        guessWord, strikes, guesses = generateUnder(guessWord, word, letter, count, strikes, guesses)
        print('You have', strikes, 'strikes.')
        print('You have currently guessed: ', guesses)
        return count, guessWord, strikes, word, guesses


while True: #using while True because for some reason strikes <=5 or stringedWord == word[:-1] doesn't work
    count, guessWord, strikes, word, guesses = play(count, guessWord, strikes, word, guesses)
    stringedWord = ''.join(guessWord)
    if strikes > 5:
        print('You lose! Sorry...')
        break
    if stringedWord == word[:-1]:
        print('You win! Congratulations!! :D')
        break

