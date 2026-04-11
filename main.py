import os 
import numpy as np
import matplotlib.pyplot as plt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import argparse
from scipy.optimize import curve_fit
from numba import njit

def parser_arguments():
    
    """
    Funzione che definisce gli argomenti da passare quando si esegue il codice
    """

    parser = argparse.ArgumentParser(description='Simulazione dei moti di particelle in scatola 2D', usage ='python3 main.py --option')
    
    parser.add_argument('-N', '--part', type=int, action='store', default=200, help='Inserisci il numero di particelle (Default: 200)')
    parser.add_argument('-v', '--vel', type=float, action='store', default=1.0, help='Inserisci il valore della velocità iniziale (default: 1.0)')    
    parser.add_argument('-t', '--tmax', type=float, action='store', default=60.0, help='Tempo massimo della simulazione (default: 60.0)')
    
    return parser.parse_args()


def maxwell_2d(v, T):
    return (v / T) * np.exp(-v**2 / (2*T))


@njit
def step(pos, vel, L, radius, dt):
    N = pos.shape[0]

    pos += vel * dt

    # Pareti
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
    
    # PYGAME
    #-----------------------
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    running = True
    t = 0.0

    # LOOP
    #-----------------------
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # calcola dt dal clock di pygame (ms -> s)
        frame_ms = clock.tick(60)
        dt_frame = frame_ms / 1000.0

        # fisica con dt variabile
        step(pos, vel, L, radius, dt_frame)

        # aggiorna tempo di simulazione e controlla t_max
        t += dt_frame
        if t >= t_max:
            running = False
        
        # DRAW
        #-----------------------
        screen.fill((0, 0, 0))

        # SIMULAZIONE
        #-----------------------
        for i in range(N):
            x = int(pos[i,0] * scale)
            y = int(pos[i,1] * scale)
            pygame.draw.circle(screen, (0,255,255), (x,y), int(radius*scale))

        # Bordo area sim
        pygame.draw.rect(screen, (255,255,255), (0,0,SIM_WIDTH,HEIGHT), 2)
        
        # ISTOGRAMMA
        #-----------------------
        speeds = np.linalg.norm(vel, axis=1)
        bins = 50 
        
        # CURVA DI MAXWELL (2D)
        #-----------------------
        T = 0.5 * np.mean(speeds**2)

        v_th = np.sqrt(2 * T)
        v_max = 5 * v_th   
        vmax_plot = v_max

        v_vals = np.linspace(0, v_max, 300)
        f_vals = maxwell_2d(v_vals, T)

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

        # Disegna assi e tick per l'istogramma
        pdf_max = max(hist.max() if hist.size>0 else 0.0, f_vals.max() if f_vals.size>0 else 0.0)
        if pdf_max <= 0:
            pdf_max = 1e-6

        n_xticks = 5

        # Asse X: velocità (tick e valori)
        for k in range(n_xticks + 1):
            v_tick = k * (v_max / n_xticks)
            x_tick = int(SIM_WIDTH + v_tick * scale_x)
            pygame.draw.line(screen, (255,255,255), (x_tick, HEIGHT-6), (x_tick, HEIGHT), 1)
            lbl = font.render(f"{v_tick:.2f}", True, (255,255,255))
            screen.blit(lbl, (x_tick - lbl.get_width()//2, HEIGHT - 22))

        # Titoli assi
        title_x = font.render("v (velocità)", True, (255,255,255))
        screen.blit(title_x, (SIM_WIDTH + HIST_WIDTH//2 - title_x.get_width()//2, HEIGHT - 40))

        # Velocità media
        mean_speed = np.mean(speeds)
        mean_text = font.render(f"Vel. media: {mean_speed:.3f}", True, (255,255,255))
        screen.blit(mean_text, (SIM_WIDTH + 50, 40))

        # Titolo e tempo
        text = font.render("Distribuzione velocità", True, (255,255,255))
        screen.blit(text, (SIM_WIDTH + 50, 10))

        time_text = font.render(f"t = {t:.1f} s", True, (255,255,255))
        text_rect = time_text.get_rect()
        text_rect.topright = (SIM_WIDTH - 10, 10)
        screen.blit(time_text, text_rect)
        
        pygame.display.flip()
    
    pygame.quit()
    analyze_statistics(vel, t)
    plt.show()

    
def analyze_statistics(vel, t):
    
    # Statistiche teoriche (dalle condizioni iniziali <v^2> = v0^2)
    T_theoretical = 0.5 * (v0**2)
    v_mp_theoretical = np.sqrt(T_theoretical)
    v_mean_theoretical = np.sqrt(np.pi * T_theoretical / 2)
    energy_per_particle_theoretical = 0.5 * v0**2
    total_energy_theoretical = energy_per_particle_theoretical * N

    # Statistiche finali simulazione
    final_speeds = np.linalg.norm(vel, axis=1)
    energy_per_particle = 0.5 * np.mean(final_speeds**2)
    std_v2 = np.std(final_speeds**2)
    total_energy = energy_per_particle * N
    T_final = energy_per_particle
    T_err = 0.5 * std_v2 / np.sqrt(N)
    energy_err = T_err
    total_energy_err = N * energy_err
    mean_speed_sim = np.mean(final_speeds)
    std_speed = np.std(final_speeds)
    mean_speed_err = std_speed / np.sqrt(N)

    # Fit maxwelliana
    bins = 50
    hist, edges = np.histogram(final_speeds, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    v_vals = np.linspace(0, max(final_speeds)*1.2, 300)
    f_theoretical = maxwell_2d(v_vals, T_theoretical)
    mask = hist > 0.01  # evita rumore
    
    popt, pcov = curve_fit(maxwell_2d, centers[mask], hist[mask], p0=[T_final])
    T_fit = popt[0] 
    T_fit_err = np.sqrt(pcov[0,0])
    v_mp_fit = np.sqrt(T_fit)
    v_mp_err = 0.5 / np.sqrt(T_fit) * T_fit_err
    v_mean_fit = np.sqrt(np.pi * T_fit / 2)
    v_mean_fit_err = (np.sqrt(np.pi / 8) / np.sqrt(T_fit)) * T_fit_err
    f_fit = maxwell_2d(v_vals, T_fit)
   
    # Grafico finale del fit
    plt.figure() 
    plt.hist(final_speeds, bins=bins, density=True, alpha=0.5, label="Simulazione")
    plt.plot(v_vals, f_theoretical, 'r--', label=f"Teoria (T={T_theoretical:.3f})")
    plt.plot(v_vals, f_fit, 'k-', label=f"Fit (T={T_fit:.3f})")
    plt.xlabel("Velocità")
    plt.ylabel("PDF")
    plt.title("Distribuzione delle velocità")
    plt.legend()
    plt.grid()
    plt.show(block=False)
    
    # Stampa risultati
    print(f"\n---------------------------------------------\n")
    print(f"--- Simulazione terminata ---\n")
    print(f"Tempo totale simulazione: {t:.1f} s")
    print(f"Numero di particelle: {N}")
    print(f"Velocità iniziale: {v0:.3f}")
    print(f"\n---------------------------------------------\n")
    print("--- Valori teorici attesi ---\n")
    print(f"Velocità media: {v_mean_theoretical:.3f}")
    print(f"Velocità più probabile: {v_mp_theoretical:.3f}")
    print(f"Energia media per particella: {energy_per_particle_theoretical:.3f}")
    print(f"Energia totale: {total_energy_theoretical:.3f}")
    print(f"Temperatura: {T_theoretical:.3f}")
    print(f"\n---------------------------------------------\n")
    print(f"--- Statistiche simulazione ---\n")
    print(f"Velocità media: {mean_speed_sim:.3f} ± {mean_speed_err:.3f}")
    print(f"Energia media per particella: {energy_per_particle:.3f} ± {energy_err:.3f}")
    print(f"Energia totale: {total_energy:.3f} ± {total_energy_err:.3f}")
    print(f"Temperatura: {T_final:.3f} ± {T_err:.3f}\n")
    print(f"\n---------------------------------------------\n")
    print(f"--- Statistiche fit ---\n")
    print(f"Velocità media: {v_mean_fit:.3f} ± {v_mean_fit_err:.3f}")
    print(f"Velocità più probabile: {v_mp_fit:.3f} ± {v_mp_err:.3f}")
    print(f"Temperatura: {T_fit:.3f} ± {T_fit_err:.3f}\n")
    print(f"--- ERRORI RELATIVI ---\n")
    print(f"Errore energia: {(np.abs(energy_per_particle - energy_per_particle_theoretical)/energy_per_particle_theoretical):.2%}")
    print(f"Errore T (simulazione): {(np.abs(T_final - T_theoretical)/T_theoretical):.2%}")
    print(f"Errore T (fit):   {(np.abs(T_fit - T_theoretical)/T_theoretical):.2%}")
    print(f"\n---------------------------------------------\n")

    return


if __name__ == "__main__":

    # Richiamo della funzione degli argomenti
    #-----------------------
    args = parser_arguments()    
    
    # PARAMETRI
    #-----------------------
    N = args.part
    L = 10.0
    radius = 0.05
    dt = 0.01

    WIDTH = 1000
    HEIGHT = 600

    SIM_WIDTH = 600
    HIST_WIDTH = 400

    scale = SIM_WIDTH / L
    
    # INIZIALIZZAZIONE
    #-----------------------
    pos = np.random.rand(N, 2) * L

    v0 = args.vel
    angles = np.random.rand(N) * 2*np.pi
    vel = np.column_stack((v0*np.cos(angles), v0*np.sin(angles)))
    t_max = args.tmax

    simulation()