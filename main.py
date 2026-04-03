import sys
import numpy as np
import pygame
import argparse
from numba import njit

def parser_arguments():
    
    """
    Funzione che definisce gli argomenti da passare quando si esegue il codice
    """

    parser = argparse.ArgumentParser(description='Simulazione dei moti di particelle in scatola 2D', usage ='python3 main.py --option')
    
    parser.add_argument('-N', '--part', type=int, action='store', default=200, help='Inserisci il numero di particelle (Default: 200)')
    parser.add_argument('-v', '--vel', type=float, action='store', default=1.0, help='Inserisci il valore della velocità iniziale (default: 1.0)')    
    
    return parser.parse_args()

# ======================
# NUMBA STEP
# ======================
@njit
def step(pos, vel, L, radius, dt):
    N = pos.shape[0]

    pos += vel * dt

    # pPareti
    for i in range(N):
        for d in range(2):
            if pos[i,d] < radius:
                pos[i,d] = radius
                vel[i,d] *= -1
            elif pos[i,d] > L - radius:
                pos[i,d] = L - radius
                vel[i,d] *= -1

    # Collisioni
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist2 = dx*dx + dy*dy

            if dist2 < (2*radius)**2 and dist2 > 1e-12:
                dist = np.sqrt(dist2)

                nx = dx / dist
                ny = dy / dist

                dvx = vel[i,0] - vel[j,0]
                dvy = vel[i,1] - vel[j,1]

                v_rel = dvx*nx + dvy*ny

                if v_rel < 0:
                    vel[i,0] -= v_rel * nx
                    vel[i,1] -= v_rel * ny
                    vel[j,0] += v_rel * nx
                    vel[j,1] += v_rel * ny

def simulation():
    # ======================
    # PYGAME
    # ======================
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    running = True

    # ======================
    # LOOP
    # ======================
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fisica
        step(pos, vel, L, radius, dt)

        # ======================
        # DRAW
        # ======================
        screen.fill((0, 0, 0))

        # --- SIMULAZIONE ---
        for i in range(N):
            x = int(pos[i,0] * scale)
            y = int(pos[i,1] * scale)
            pygame.draw.circle(screen, (0,255,255), (x,y), int(radius*scale))

        # Bordo area sim
        pygame.draw.rect(screen, (255,255,255), (0,0,SIM_WIDTH,HEIGHT), 2)

        # ======================
        # ISTOGRAMMA
        # ======================
        speeds = np.linalg.norm(vel, axis=1)
        bins = 50
        
        # ======================
        # CURVA DI MAXWELL (2D)
        # ======================
        T = 0.5 * np.mean(speeds**2)

        v_th = np.sqrt(2 * T)
        v_max = 5 * v_th   
        vmax_plot = v_max

        v_vals = np.linspace(0, v_max, 300)
        f_vals = (v_vals / T) * np.exp(-v_vals**2 / (2*T))

        #vmax_plot = 3 * np.sqrt(np.mean(speeds**2))  # scala adattiva
        hist, edges = np.histogram(speeds, bins=bins, range=(0, vmax_plot), density=True)
        
        scale_y = HEIGHT - 50
        scale_x = HIST_WIDTH / v_max

        for i in range(bins):
            x0 = SIM_WIDTH + i * (HIST_WIDTH // bins)
            width = HIST_WIDTH // bins

            height = int(hist[i] * scale_y)

            y0 = HEIGHT - height

            pygame.draw.rect(screen, (100,200,100), (x0, y0, width-2, height))

        # Disegna curva di Maxwell
        points = []

        for i in range(len(v_vals)):
            y = HEIGHT - f_vals[i] * scale_y
            x = SIM_WIDTH + v_vals[i] * scale_x

            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(screen, (255, 50, 50), False, points, 2)

        # Titolo
        text = font.render("Distribuzione velocità", True, (255,255,255))
        screen.blit(text, (SIM_WIDTH + 50, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":

    # Richiamo della funzione degli argomenti
    args = parser_arguments()
        
    # ======================
    # PARAMETRI
    # ======================
    N = args.part
    L = 10.0
    radius = 0.05
    dt = 0.01

    WIDTH = 1000
    HEIGHT = 600

    SIM_WIDTH = 600
    HIST_WIDTH = 400

    scale = SIM_WIDTH / L

    # ======================
    # INIZIALIZZAZIONE
    # ======================
    pos = np.random.rand(N, 2) * L

    v0 = args.vel
    angles = np.random.rand(N) * 2*np.pi
    vel = np.column_stack((v0*np.cos(angles), v0*np.sin(angles)))
        
    simulation()