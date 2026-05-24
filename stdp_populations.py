"""
Neural populations connected with a spike-timing-dependent plasticity (STDP) synapse
"""
import numpy as np
import pyNN.spiNNaker as sim
import matplotlib.pyplot as plt
from pyNN.standardmodels.synapses import STDPMechanism
from pyNN.utility.plotting import Figure, Panel
from pyNN.space import RandomStructure, Sphere

def main():
    sim.setup(timestep=1.0)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

    # Parameters of neuron model; most of these are the default parameters and are listed for readability
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

    runtime = 250  # Simulate for 250 milliseconds

    neuron = sim.IF_curr_exp(**neuron_parameters)

    number_of_neurons = 1000

    # Pre-synaptic population
    pre_population = sim.Population(
        number_of_neurons,
        neuron,
        structure=RandomStructure(boundary=Sphere(radius=150)),
        initial_values={"v": -70.0},
        label="Pre-Synaptic Population of IF_curr_exp Neurons"
    )

    # Post-synaptic population
    post_population = sim.Population(
        number_of_neurons,
        neuron,
        structure=RandomStructure(boundary=Sphere(radius=150)),
        initial_values={"v": -70.0},
        label="Post-Synaptic Population of IF_curr_exp Neurons"
    )

    stdp_synapse = sim.STDPMechanism(
        weight=2.5,
        delay="0.2 + 0.01*d",
        timing_dependence=sim.SpikePairRule(
            tau_plus=20.0,
            tau_minus=20.0,
            A_plus=0.01,
            A_minus=0.012
        ),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=0.0,
            w_max=0.04
        )
    )

    connector = sim.OneToOneConnector()

    sim.Projection(
        presynaptic_population=pre_population,
        postsynaptic_population=post_population,
        connector=connector,
        synapse_type=stdp_synapse,
        receptor_type="excitatory",
        label="Excitatory Connections"
    )

    # Noisy Current Source
    current = sim.NoisyCurrentSource(mean=1.5, stdev=1.0, start=50.0, stop=runtime * 0.85, dt=1.0)
    pre_population[:number_of_neurons].inject(current)

    pre_population.record("v")
    post_population.record("v")

    sim.run(runtime)

    v_data_pre_neuron = pre_population.get_data().segments[0].filter(name="v")[0][:, 0]  # Get the data for the first neuron in pre-synaptic population
    v_data_post_neuron = post_population.get_data().segments[0].filter(name="v")[0][:, 0]  # Get the data for the first neuron in post-synaptic population

    Figure(
        Panel(v_data_pre_neuron,
              ylabel="Membrane Potential (mV)",
              xlabel="Time (ms)",
              data_labels=[pre_population.label],
              yticks=True,
              xticks=True,
              xlim=(0, runtime),
              ylim=(-90, 60)),
        Panel(v_data_post_neuron,
              ylabel="Membrane Potential (mV)",
              xlabel="Time (ms)",
              data_labels=[post_population.label],
              yticks=True,
              xticks=True,
              xlim=(0, runtime),
              ylim=(-90, 60)),
        
        title="LIF Model with Spike-Timing-Dependent Plasticity (STDP) and DEPSC",
        annotations=f"Simulated with {sim.name()}",
    ).save("STDP_populations.png")

    sim.end()
    plt.show()

if __name__ == "__main__":
    main()