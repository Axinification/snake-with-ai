import pygame
import math
import random
import tkinter as tk
from tkinter import messagebox

pygame.init()

class cube(object):
	size = 500
	rows = 25
	def __init__(self, start, direction_x=0, direction_y=0, color=(255,0,0)):
		self.position = start
		self.direction_x = 0
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
		if eyes:
			centre = distance//2
			radius = 3
			circleMiddle = (x*distance+centre-radius, y*distance+8)
			circleMiddle2 = (x*distance+distance-radius*2, y*distance+8)
			pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
			pygame.draw.circle(surface, (0,0,0), circleMiddle2, radius)

class snake(object):
	body = []
	turns = {}

	def __init__(self, color, position):
		self.color = color
		self.head = cube(position)
		self.body.append(self.head)
		self.direction_x = 1
		self.direction_y = 0

	def move(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

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

		for i, c in enumerate(self.body):
			p = c.position[:]
			if p in self.turns:
				turn = self.turns[p]
				c.move(turn[0],turn[1])
				if i == len(self.body)-1:
					self.turns.pop(p)
		#	else:
		#		#EASY MODE - no walls
		#		if c.direction_x == -1 and c.position[0] <= 0: c.position = (c.rows-1, c.position[1])
		#		elif c.direction_x == 1 and c.position[0] >= c.rows-1: c.position = (0, c.position[1])
		#		elif c.direction_y == -1 and c.position[1] <= 0: c.position = (c.position[0], c.rows-1)
		#		elif c.direction_y == 1 and c.position[1] >= c.rows-1: c.position = (c.position[0], 0)
		#		else: c.move(c.direction_x,c.direction_y) 
			else:
				#NORMAL MODE - walls
				if c.direction_x == -1 and c.position[0] <= 0 or c.direction_x == 1 and c.position[0] >= c.rows-1 or c.direction_y == -1 and c.position[1] <= 0 or c.direction_y == 1 and c.position[1] >= c.rows-1:
					score = str(len(s.body))
					message('!You Lost!', 'Score: ' + score)
					s.reset((10,10))
					break
				else: c.move(c.direction_x,c.direction_y)


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

class line(object):
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

		self.angle = math.degrees(math.atan2(distance_y,distance_x))

		pygame.draw.line(surface, (155,255,155), (self.head_cords), (self.snack_cords))

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

def message(subject, content):
	root = tk.Tk()
	root.attributes("-topmost", True)
	root.withdraw()
	messagebox.showinfo(subject, content)
	try:
		root.destroy()
	except:
		pass

def main():
	global size, rows, s, snack, line
	size = 500
	rows = 25
	window = pygame.display.set_mode((size, size))
	s = snake((255,0,0), (10,10))
	snack = cube(randomSnack(rows,s), color=(0,255,0))
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
			snack = cube(randomSnack(rows,s), color=(random.randrange(255),random.randrange(255),random.randrange(255)))
		
		for x in range(len(s.body)):
			if s.body[x].position in list(map(lambda z:z.position, s.body[x+1:])):
				score = str(len(s.body))
				message('!You Lost!', 'Score: ' + score)
				s.reset((10,10))
				break
		
		redrawWindow(window)
	pass

main()

