"""
Leaky integrate and fire (LIF) neuron population with decaying-exponential post-synaptic current
"""


import pyNN.spiNNaker as sim
from neo import AnalogSignal
from pyNN.space import RandomStructure, Sphere
import numpy as np
import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel
from quantities import mV, ms

def main():
    sim.setup(timestep=1.0)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

    # Parameters of neuron model; these are the default parameters and are written for readability
    neuron_parameters = {
        "v_rest": -65.0,
        "cm": 1.0,
        "tau_m": 20.0,
        "tau_refrac": 0.1,
        "tau_syn_E": 5.0,
        "tau_syn_I": 5.0,
        "i_offset": 0.0,
        "v_reset": -65.0,
        "v_thresh": -50.0
    }

    runtime = 50 # Simulate for 50 milliseconds

    neuron = sim.IF_curr_exp(**neuron_parameters)

    number_of_neurons = 1000
    population = sim.Population(
        number_of_neurons,
        neuron,
        structure=RandomStructure(boundary=Sphere(radius=150)),
        initial_values={"v": -70.0},
        label="IF_curr_exp"
    )

    time = np.arange(0.0, runtime, 1.0)
    amplitudes = 0.1 * np.sin(time * np.pi / 100.0)

    current = sim.StepCurrentSource(times=time.tolist(), amplitudes=amplitudes.tolist())
    population[:200].inject(current)

    population.record("v")

    sim.run(runtime)
    v_data = population.get_data().segments[0].filter(name="v")[0][:, number_of_neurons - 1] # Get the data for the last neuron
    # v_data_mean = AnalogSignal(v_data.mean(axis=1).reshape(runtime, 1)*mV, sampling_period=runtime*ms) # For viewing the average among several neurons

    Figure(
        Panel(v_data,
              ylabel="Membrane Potential (mV)",
              xlabel="Time (ms)",
              data_labels=[population.label],
              yticks=True,
              xticks=True,
              xlim=(0, runtime)),
        title="Leaky Integrate and Fire (LIF) Model",
        annotations=f"Simulated with {sim.name()}",
    ).save("LIF.png")

    sim.end()
    plt.show()

if __name__ == "__main__":
    main()