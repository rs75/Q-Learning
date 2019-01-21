import time, threading
from tkinter import *
import numpy as np

#responsoble for the GUI
class App(object):
    def __init__(self, root):
        self.width = 600
        self.height = 600

        # root.geometry(200,200)
        root.minsize(width=self.width, height=self.height)
        root.maxsize(width=self.width, height=self.height)
        root.title("REINFORCEMENT LEARNING")
        # l1 = Label(root, text="Hello Tkinter!")
        # l1.pack(side = 'left')
        self.w = Canvas(root, width=self.width, height=self.height)
        self.w.pack()
        self.last_field = np.zeros((10, 10))
        self.last_field[:] = -1
        self.last_player_pos = (0, 0)
        self.Q = None

    def update_canvas(self, field, player_pos):
        self.last_field[self.last_player_pos[1], self.last_player_pos[0]] = -1
        # self.w.delete('all')
        for y in range(len(field)):
            for x in range(len(field[y])):
                if self.last_field[y, x] != field[y][x]:
                    self.last_field[y, x] = field[y][x].copy()
                    if field[y][x] == 0:
                        self.w.create_rectangle(x * (self.width / 10), y * (self.height / 10), (x + 1) * (self.width / 10), (y + 1) * (self.height / 10), fill="#eeeeee")
                    elif field[y][x] == 1:
                        self.w.create_rectangle(x * (self.width / 10), y * (self.height / 10), (x + 1) * (self.width / 10), (y + 1) * (self.height / 10), fill="#333333")
                    elif field[y][x] == 2:
                        self.w.create_rectangle(x * (self.width / 10), y * (self.height / 10), (x + 1) * (self.width / 10), (y + 1) * (self.height / 10), fill="#00a200")
                    elif field[y][x] == 3:
                        self.w.create_rectangle(x * (self.width / 10), y * (self.height / 10), (x + 1) * (self.width / 10), (y + 1) * (self.height / 10), fill="#44a222")
        self.w.create_rectangle(player_pos[0] * (self.width / 10), player_pos[1] * (self.height / 10), (player_pos[0] + 1) * (self.width / 10), (player_pos[1] + 1) * (self.height / 10), fill="#007eff")

        state = len(field[0]) * self.last_player_pos[1] + self.last_player_pos[0]
        # max_res = max(self.Q[state])
        max_res = self.Q[state]
        max_res = np.round(max_res, 3)

        max = np.max(max_res)
        # print(max_res)

        self.w.create_text((self.last_player_pos[0] + 0.25) * (self.width / 10), (self.last_player_pos[1] + 0.4) * (self.height / 10), text='' + str(max_res[0]), fill="#000000", font=("Purisa", 8))
        self.w.create_text((self.last_player_pos[0] + 0.65) * (self.width / 10), (self.last_player_pos[1] + 0.6) * (self.height / 10), text='' + str(max_res[1]), fill="#000000", font=("Purisa", 8))
        self.w.create_text((self.last_player_pos[0] + 0.5) * (self.width / 10), (self.last_player_pos[1] + 0.1) * (self.height / 10), text='' + str(max_res[2]), fill="#000000", font=("Purisa", 8))
        self.w.create_text((self.last_player_pos[0] + 0.5) * (self.width / 10), (self.last_player_pos[1] + 0.8) * (self.height / 10), text='' + str(max_res[3]), fill="#000000", font=("Purisa", 8))
        self.w.create_text((self.last_player_pos[0] + 0.5) * (self.width / 10), (self.last_player_pos[1] + 0.3) * (self.height / 10), text='' + str(max), fill="#ff0000", font=("Purisa", 8))

        self.last_player_pos = player_pos


class game():
    # static
    learning_rate = 0.8
    actions = [0, 1, 2, 3]
    num_actions = 4
    num_states = 100
    count = 0
    learn_until = 1000#8000
    Q = np.zeros((num_states, num_actions))

    # actions = left , right , up , down = 0,1,2,3
    def __init__(self, window):
        self.window = window
        window.Q = game.Q
        self.field = np.zeros((10, 10))
        mask = self.field[1:-1, 1:-1] < 0
        mask_all = self.field >= 0
        mask_all[1:-1, 1:-1] = mask
        self.field[mask_all] = 1
        inside = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 3]
        ]
        self.field[1:-1, 1:-1] = inside
        self.player_pos_y = 1
        self.player_pos_x = 1

        if game.count > game.learn_until:# or True:
           self.window.update_canvas(self.field, (self.player_pos_x, self.player_pos_y))

        #if game.count>5000:
        #    self.field[3,1] = 1

    #returns action with best expected reward
    def get_action(self, state):
        q = np.array([game.Q[state, a] for a in game.actions])
        max = np.max(q)
        max_q = np.argmax(q)

        # bei mehreren max val  zufällig aussuchen
        all_best = np.argwhere(q == max)
        arg_max_q = all_best[np.random.randint(0, len(all_best))]

        return arg_max_q[0]

    def next(self, random_probability):
        self.state = len(self.field[0]) * self.player_pos_y + self.player_pos_x
        action = self.get_action(self.state)
        # random


        # if game.count < game.learn_until:
        if np.random.randint(0, 101) < random_probability:
            action = np.random.randint(0, 4)
        if game.count % 100 == 0:
            print(game.count, 'iterations')

        last_pos = (self.player_pos_x,self.player_pos_y)
        if action == 0:
            self.player_pos_x -= 1
        elif action == 1:
            self.player_pos_x += 1
        elif action == 2:
            self.player_pos_y -= 1
        elif action == 3:
            self.player_pos_y += 1

        if game.count > game.learn_until:# or True:
           self.window.update_canvas(self.field, (self.player_pos_x, self.player_pos_y))

        award = 0
        winner = 0  # 1 lose , 2 win

        if self.field[self.player_pos_y, self.player_pos_x] == 1:
            award -= 300
            self.player_pos_x = last_pos[0]
            self.player_pos_y=last_pos[1]
            winner = 0
        elif self.field[self.player_pos_y, self.player_pos_x] == 2:
            award += 10
            winner = 2
        elif self.field[self.player_pos_y, self.player_pos_x] == 3:
            award += 3000
            winner = 2
        elif self.field[self.player_pos_y, self.player_pos_x] == 0:
            award  -=1

        next_state = len(self.field[0]) * self.player_pos_y + self.player_pos_x
        #reward for best next state
        max_next = max(game.Q[next_state])
        # print( game.Q[self.state,action],type(game.Q[self.state,action]),self.state,type(self.state),action,type(action))
        # print(game.Q[self.state,action] +  (game.learning_rate * (award+max_next)))

        old_q = game.Q[self.state, action]
        game.Q[self.state, action] = old_q + (game.learning_rate * (award + 0.95*(max_next - old_q)))
        # print("-------")
        # print(game.Q)

        game.count += 1

        return winner


class Simulator(threading.Thread):
    def __init__(self, window):
        threading.Thread.__init__(self)
        self.window = window
        #self.random_probability = 0.5

        self.wait_seconds = 0.5


    def run(self):
        while 1:
            # main programm

            if game.count < game.learn_until:
                self.random_probability  = 0.3 - (0.01 * (game.count/100) )
            else:
                self.random_probability =0
            self.random_probability =  np.max((0,self.random_probability))

            #simulate new game
            g = game(self.window)
            while 1:
                if game.count< game.learn_until:
                    time.sleep(0)
                else:
                    time.sleep(self.wait_seconds)
                res = g.next(self.random_probability) # next action
                if res == 1:
                    # lose
                    break
                if res == 2:
                    # win
                    break


if __name__ == '__main__':
    mainWindow = Tk()
    app = App(mainWindow)

    Process = Simulator(app)
    Process.start()

    mainWindow.mainloop()
    Process.join()
