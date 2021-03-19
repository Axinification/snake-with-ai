# import pygame
# import math
# import random
# import tkinter as tk
# from tkinter import messagebox
# from enum import Enum

import pygame
import random
import math
from enum import Enum
from collections import namedtuple

pygame.init()

#VARIABLES
SIZE = 500
ROWS = 25
SNAKE_COLOR = (255,0,0)
SNAKE_POSITION = (10,10)
RANDOM_COLOR = (random.randrange(255),random.randrange(255),random.randrange(255))




class cube(object):
	size = 500
	rows = 25
	def __init__(self, start, direction_x=0, direction_y=0, color=(255,0,0)):
		self.position = start
		#Here you can set the direction the snake will be moving at the start
		self.direction_x = 1
		self.direction_y = 0
		self.color = color

	def move(self, direction_x, direction_y):
		self.direction_x = direction_x
		self.direction_y = direction_y
		self.position = (self.position[0] + self.direction_x, self.position[1] + self.direction_y)

	def draw(self, surface, eyes=False):
		distance = self.size // self.rows
		x = self.position[0]
		y = self.position[1]
		pygame.draw.rect(surface, self.color, (x*distance+1, y*distance+1, distance-2, distance-2))
		if eyes: # Adding eyes
			centre = distance//2
			radius = 3
			circleMiddle = (x*distance+centre-radius, y*distance+8)
			circleMiddle2 = (x*distance+distance-radius*2, y*distance+8)
			pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
			pygame.draw.circle(surface, (0,0,0), circleMiddle2, radius)

class snake(object):
	body = [] #Array for body parts
	turns = {} #Dictionary for position of body parts

	def __init__(self, color, position):
		self.color = color
		self.head = cube(position)
		self.body.append(self.head)
		#Here you can set the direction the snake will be facing during first move on left right version, 
		#[0:0] works only for wasd version
		self.direction_x = 0
		self.direction_y = 0

	def move(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				
			#WASD Version - 4 inputs
			keys = pygame.key.get_pressed()
			for key in keys:
				if keys[pygame.K_LEFT] and self.direction_x != 1:
					self.direction_x = -1
					self.direction_y = 0
					self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]

				elif keys[pygame.K_RIGHT] and self.direction_x != -1:
					self.direction_x = 1
					self.direction_y = 0
					self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]

				elif keys[pygame.K_UP] and self.direction_y != 1:
					self.direction_x = 0
					self.direction_y = -1
					self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]

				elif keys[pygame.K_DOWN] and self.direction_y != -1:
					self.direction_x = 0
					self.direction_y = 1
					self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]

			#Left Right Version - 2 inputs
			# if snake is set to not be moving, left will make it go down and right up
			# if event.type == pygame.KEYDOWN:
			# 	if event.key == pygame.K_LEFT:
			# 		if self.direction_x == 1 and self.direction_y == 0:
			# 			self.direction_x = 0
			# 			self.direction_y = -1
			# 			self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]
			# 		elif self.direction_x == 0 and self.direction_y == -1:
			# 			self.direction_x = -1
			# 			self.direction_y = 0
			# 			self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]
			# 		elif self.direction_x == -1 and self.direction_y == 0:
			# 			self.direction_x = 0
			# 			self.direction_y = 1
			# 			self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]
			# 		elif self.direction_x == 0 and self.direction_y == 1:
			# 			self.direction_x = 1
			# 			self.direction_y = 0
			# 			self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]
			# 	elif event.key == pygame.K_RIGHT:
					if self.direction_x == 1 and self.direction_y == 0:
						self.direction_x = 0
						self.direction_y = 1
						self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]
					elif self.direction_x == 0 and self.direction_y == 1:
						self.direction_x = -1
						self.direction_y = 0
						self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]
					elif self.direction_x == -1 and self.direction_y == 0:
						self.direction_x = 0
						self.direction_y = -1
						self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]
					elif self.direction_x == 0 and self.direction_y == -1:
						self.direction_x = 1
						self.direction_y = 0
						self.turns[self.head.position[:]] = [self.direction_x, self.direction_y]

		#Create a list of moves on saved positions
		for interation, cube in enumerate(self.body):
			position = cube.position[:]
			if position in self.turns:
				turn = self.turns[position]
				cube.move(turn[0],turn[1])
				if interation == len(self.body)-1: #If enumeration list is longer than snake body, pop last saved turn
					self.turns.pop(position)
		#	else:
		#		#EASY MODE - no walls
		#		if c.direction_x == -1 and c.position[0] <= 0: c.position = (c.rows-1, c.position[1])
		#		elif c.direction_x == 1 and c.position[0] >= c.rows-1: c.position = (0, c.position[1])
		#		elif c.direction_y == -1 and c.position[1] <= 0: c.position = (c.position[0], c.rows-1)
		#		elif c.direction_y == 1 and c.position[1] >= c.rows-1: c.position = (c.position[0], 0)
		#		else: c.move(c.direction_x,c.direction_y) 
			else:
				#NORMAL MODE - walls
				#Collision detection
				if (cube.direction_x == -1 and cube.position[0] <= 0 or 
					cube.direction_x == 1 and cube.position[0] >= cube.rows-1 or 
					cube.direction_y == -1 and cube.position[1] <= 0 or 
					cube.direction_y == 1 and cube.position[1] >= cube.rows-1):
					#Print the score
					score = str(len(s.body))
					# message('!You Lost!', 'Score: ' + score)
					s.reset((10,10))
					break
				else: cube.move(cube.direction_x,cube.direction_y)


	def reset(self, position):
		self.head = cube(position)
		self.body = []
		self.body.append(self.head)
		self.turns = {}
		self.direction_x = 0
		self.direction_y = 0
	
	def addCube(self):
		tail = self.body[-1]
		dx, dy = tail.direction_x, tail.direction_y

		if dx == 1 and dy == 0:
			self.body.append(cube((tail.position[0]-1, tail.position[1])))
		elif dx == -1 and dy == 0:
			self.body.append(cube((tail.position[0]+1, tail.position[1])))
		elif dx == 0 and dy == 1:
			self.body.append(cube((tail.position[0], tail.position[1]-1)))
		elif dx == 0 and dy == -1:
			self.body.append(cube((tail.position[0], tail.position[1]+1)))

		self.body[-1].direction_x = dx
		self.body[-1].direction_y = dy

	def draw(self, surface):
		for i, c in enumerate(self.body):
			if i == 0:
				c.draw(surface, True)
			else:
				c.draw(surface)

class line(object): #class for drawing the helper line and angle calculationg
	global rows, size, s, snack
	
	def __init__(self):
		self.angle = 0.0
		self.head_cords = (0,0)
		self.snack_cords = (0,0)
		self.distance = size // rows

	def draw(self, surface):
		self.head_cords = (s.head.position[0]*self.distance,s.head.position[1]*self.distance)
		self.snack_cords = (snack.position[0]*self.distance,snack.position[1]*self.distance)
		distance_x = self.snack_cords[0] - self.head_cords[0]
		distance_y = self.snack_cords[1] - self.head_cords[1]

		#Calculating the angle
		self.angle = math.degrees(math.atan2(distance_y,distance_x))
		#Helper line for angle
		pygame.draw.line(surface, (155,255,155), (self.head_cords), (self.snack_cords))
		#Setting up the helper angle calculation in the upper left corner
		font = pygame.font.SysFont('Arial', 18, bold=False)
		label = font.render('Angle: ' + str(self.angle), False , (255,255,255))
		surface.blit(label, (10,0))


def drawGrid(size, rows, surface):
	sizeBetween = size // rows
	x = 0
	y = 0
	for i in range(rows):
		x += sizeBetween
		y += sizeBetween

		#Draw grid
		pygame.draw.line(surface, (255,255,255), (x,0), (x,size))
		pygame.draw.line(surface, (255,255,255), (0,y), (size,y))

def redrawWindow(surface):
	global rows, size, s, snack, line
	surface.fill((0,0,0))
	s.draw(surface)
	snack.draw(surface)
	line.draw(surface)
	drawGrid(size, rows, surface)
	pygame.display.update()

def randomSnack(r, item):
	global rows
	positions = item.body

	while True:
		x = random.randrange(rows)
		y = random.randrange(rows)
		if len(list(filter(lambda z:z.position == (x,y), positions))) > 0:
			continue
		else:
			break
	return (x,y)

# def message(subject, content):
# 	root = tk.Tk()
# 	root.attributes("-topmost", True)
# 	root.withdraw()
# 	messagebox.showinfo(subject, content)
# 	try:
# 		root.destroy()
# 	except:
# 		pass

def main():
	global size, rows, s, snack, line
	size = SIZE
	rows = ROWS
	window = pygame.display.set_mode((size, size))
	s = snake(SNAKE_COLOR, SNAKE_POSITION)
	snack = cube(randomSnack(rows,s), color=RANDOM_COLOR)
	line = line()
	flag = True
	clock = pygame.time.Clock()

	while flag:

	#Speed calcultaion
		pygame.time.delay(50) #Pause for 50 miliseconds each frame
		clock.tick(10) #Run the game at 10 fps
		s.move() #Move the snak in the given direction
		if s.body[0].position == snack.position: # If snake head is on the same position as snack, remove the snack, gen
			s.addCube()
			snack = cube(randomSnack(rows,s), color=RANDOM_COLOR)
		
		for x in range(len(s.body)):
			if s.body[x].position in list(map(lambda z:z.position, s.body[x+1:])):
				# score = str(len(s.body))
				# message('!You Lost!', 'Score: ' + score)
				s.reset((10,10))
				break
		
		redrawWindow(window)
	pass


main()

