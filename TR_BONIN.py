from random import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


#### Defining classes that are needed to simulate create the supply chain  


class Product:  # All the products used in the supply chain (fished products + intermediary) are created with this class
    def __init__(self, product_dictionnary):
        self.product_dict = product_dictionnary.copy()


class Unlimitied_stock: # Raw material stock
    def __init__(self, raw_material_dictionnary):
        self.raw_material_dict = raw_material_dictionnary.copy()
        self.stock_out = self.raw_material_dict["stock_out_ini"].copy()

    def reset(self):
        self.stock_out = self.raw_material_dict["stock_out_ini"].copy()


class Store:
    def __init__(self, store_dictionnary):
        self.store_dict = store_dictionnary.copy()
        self.stock_in = self.store_dict["stock_in_ini"] # Store stock
        self.transits = []  # Products on transit in trucks from a stock to the store

    def reset(self):
        self.stock_in = self.store_dict["stock_in_ini"]
        self.transits = []

    def placing_order(self, order): # Ordering items from the upstream stock, according to agent action "order"
        quantity = order * 2 / 10 * discrete_ordering_pace
        delivery_time = np.random.poisson(self.store_dict["delivery_parameters"])
        arriving_stock = min(
            self.store_dict["source"].stock_out[
                self.store_dict["product_stock_in_name"]
            ],
            quantity,
        )
        self.transits.append((arriving_stock, delivery_time))   # Items loaded on trucks are now on transit towards the store: (quantity, time)
        self.store_dict["source"].stock_out[
            self.store_dict["product_stock_in_name"]
        ] -= arriving_stock # Removing items from source stock

    def step(self, demand):
        new_transit = list()
        missed_sales = 0
        for quantity, remain_time in self.transits: # Receiving items that have just arrived  
            remain_time -= 1
            if remain_time <= 0:
                self.stock_in += quantity
            else:
                new_transit.append((quantity, remain_time)) # Items not yet received
        self.transits = new_transit
        sales = min(self.stock_in, demand)
        if demand > self.stock_in:  # Computing missed sales
            missed_sales = demand - self.stock_in       
        self.stock_in -= sales
        penalty = (
            - missed_sales * self.store_dict["product_stock_in"].product_dict["no_sell_penalty"]
        )
        money_revenue = sales * self.store_dict["product_stock_in"].product_dict["sell_price"]
        return penalty, missed_sales, sales, money_revenue

    def cost_accounting(self):  # Computing stock cost
        holding_cost = (
            self.stock_in
            * self.store_dict["product_stock_in"].product_dict["storage_price"]
            * self.store_dict["cost_of_stock_multiplyer"]
        )
        return holding_cost


class Process:
    def __init__(self, process_dictionnary):
        self.process_dict = process_dictionnary.copy()
        self.stock_in = self.process_dict["stock_in_ini"]
        self.stock_out = self.process_dict["stock_out_ini"].copy()
        self.transits = []

    def reset(self):
        self.stock_in = self.process_dict["stock_in_ini"]
        self.stock_out = self.process_dict["stock_out_ini"].copy()
        self.transits = []

    def placing_order(self, order): # Same as class Store
        quantity = order * 2 / 10 * discrete_ordering_pace
        delivery_time = np.random.poisson(self.process_dict["delivery_parameters"])
        arriving_stock = min(
            self.process_dict["source"].stock_out[
                self.process_dict["product_stock_in_name"]
            ],
            quantity,
        )
        self.transits.append((arriving_stock, delivery_time))
        self.process_dict["source"].stock_out[
            self.process_dict["product_stock_in_name"]
        ] -= arriving_stock

    def step(self):
        new_transit = list()
        for quantity, remain_time in self.transits: # Same as class Store 
            remain_time -= 1
            if remain_time <= 0:
                self.stock_in += quantity
            else:
                new_transit.append((quantity, remain_time))
        self.transits = new_transit
        if random() > self.process_dict["failure_prob"]:    # If the machine does not breakdown 
            taken_from_stock = min(self.process_dict["prod_lim"], self.stock_in)
            self.stock_in -= taken_from_stock
            processed_quantity = round(
                taken_from_stock * (1 - self.process_dict["default_rate"])  # In case a production default rate must be implemented
            )
            for i in self.process_dict["stock_out_ini"]:    # Processed items are added in the downstream "stock_out" stock
                self.stock_out[i] += processed_quantity

    def cost_accounting(self):      # Computing stock costs for upstream & downstream "stock_in" and "stock_out" 
        stock_out_holding_cost = 0  # (stocks are related to either stores or processes). Similar to class Store
        stock_in_holding_cost = (
            self.stock_in
            * self.process_dict["product_stock_in"].product_dict["storage_price"]
        )
        for product in self.process_dict["products_stock_out"]:
            stock_out_holding_cost += (
                self.stock_out[product.product_dict["product_name"]]
                * product.product_dict["storage_price"]
            )
        holding_cost = (
            stock_out_holding_cost + stock_in_holding_cost
        ) * self.process_dict["cost_of_stock_multiplyer"]
        return holding_cost


class Environment:
    def __init__(self, my_environment_dictionnary):
        self.my_environment_dict = my_environment_dictionnary.copy()
        self.state, self.state_as_list = [], []
        self.holding_cost_history = []
        self.missed_sales_penalty_history = []
        self.reward_history = []
        self.missed_sales_history = []
        self.sales_history = []
        self.money_revenue_history = []
        self.step(self.my_environment_dict["first_instructions"])

    def reset(self):
        self.state, self.state_as_list = [], []
        self.holding_cost_history = []
        self.missed_sales_penalty_history = []
        self.reward_history = []
        self.missed_sales_history = []
        self.sales_history = []
        self.money_revenue_history = []
        raw_material.reset()
        for process_or_store in process_list + store_list:
            process_or_store.reset()
        self.step(self.my_environment_dict["first_instructions"])

    def step(self, agent_instruction):
        self.former_state = self.state
        self.former_state_as_list = self.state_as_list
        demands = self.demand_vector()  # Generating client demand for finished items
        holding_cost, penalty, missed_sales, sales, money_revenue = 0, 0, 0, 0, 0
        order_processes, order_stocks = (   # Recovering order amounts from agent instructions
            agent_instruction[: len(process_list)],
            agent_instruction[len(process_list) : len(process_list + store_list)],
        )
        for process_or_store, order in zip(
            process_list + store_list, order_processes + order_stocks
        ):
            process_or_store.placing_order(order)   # Processes and stores place orders
        for process in process_list:
            process.step()  # Processes produce and stores sell
        for store, demand in zip(store_list, demands):
            penalty_, missed_sales_, sales_, money_revenue_ = store.step(demand)
            penalty += penalty_
            missed_sales += missed_sales_
            sales += sales_
            money_revenue += money_revenue_
        for process_or_store in process_list + store_list:
            holding_cost += process_or_store.cost_accounting()
        self.reward = (penalty - holding_cost) / reward_div # reward_div is here to keep the reward to values close to 0 for convergence
        money_revenue -= holding_cost
        self.state = self.state_update()
        self.state_as_list, self.state_as_list_int = self.state_to_list(self.state)

        self.holding_cost_history.append(holding_cost)
        self.missed_sales_penalty_history.append(penalty)
        self.reward_history.append(self.reward)
        self.missed_sales_history.append(missed_sales)
        self.sales_history.append(sales)
        self.money_revenue_history.append(money_revenue)

        return (
            self.former_state_as_list,
            agent_instruction,
            self.reward,
            self.state_as_list,
        )

    def state_update(self):
        new_state, new_state_process, new_state_store = [], [], []
        for process in process_list:
            new_state_process.append(
                (
                    process.process_dict["process_name"],
                    process.stock_in,
                    process.stock_out,
                )
            )
        for store in store_list:
            new_state_store.append((store.store_dict["store_name"], store.stock_in))
        new_state.append(new_state_process)
        new_state.append(new_state_store)
        return new_state

    def state_to_list(self, state): # Turning "state" (list of lists) to one single list that can be used as neural network input
        state_as_list = []
        state_as_list_normal = []
        for tuple in state[0]:
            state_as_list.append(tuple[1])
            for key in tuple[2]:
                state_as_list.append(tuple[2][key])
        for tuple in state[1]:
            state_as_list.append(tuple[1])
        state_as_list_normal = state_as_list.copy()
        state_as_list_normal = np.asarray(state_as_list_normal)
        max = np.max(state_as_list_normal)
        if max != 0:
            state_as_list_normal /= max
        state_as_list_normal = state_as_list_normal.tolist()
        return state_as_list_normal, state_as_list

    def demand_vector(self):    # Generating demand
        demand = np.random.poisson(
            lam=self.my_environment_dict["demand_average"],
            size=(1, len(store_list)),
        ).tolist()[0]
        return demand


#### Defining agents 


class Agent_basic:  # Agent that orders 80 all the time
    def choose_action(self):
        instructions = []
        for _ in range(len(process_list)+len(store_list)):
            instructions.append(4)
        return instructions


class Agent_stupid: # Random agent
    def choose_action(self):
        instructions = np.random.randint(0, 10, len(process_list)+len(store_list)).tolist()
        return instructions


class Agent_NN:
    def __init__(self, agent_NN_dictionnary):
        self.agent_NN_dict = agent_NN_dictionnary.copy()
        self.epsilon = self.agent_NN_dict["epsilon"]    # Exploration rate

    def choose_action(self, state_as_list):
        actions = []
        state_tf = tf.convert_to_tensor([state_as_list], dtype=tf.float32)  # Input
        q_values = model(state_tf)  # Output
        rand = random()
        for i in range(len(q_values)):
            if rand < self.epsilon:
                choice = np.random.randint(0, len(q_values[i][0]))  # Exploration
            else:
                choice = np.argmax(q_values[i][0])
            actions.append(choice)
        if self.epsilon > 0.01:
            self.epsilon *= self.agent_NN_dict["decay_rate"]    # Reducting exploration rate
        else:
            self.epsilon = 0
        return actions, q_values

    def q_values_update(self, reward, new_q_values, instructions):
        updated_q_values = []
        for i in range(len(instructions)):
            updated = reward + self.agent_NN_dict["discount_rate"] * tf.math.reduce_max(new_q_values[i][0])
            updated_q_values.append(updated)
        return updated_q_values

    def training(self, batch):
        loss_list = []
        for state_as_list, agent_instructions, reward, new_state_as_list in batch:
            loss = tf.cast(0, dtype=tf.float32)
            _, new_q_values = self.choose_action(new_state_as_list)
            updated_q_values = self.q_values_update(
                reward, new_q_values, agent_instructions
            )
            with tf.GradientTape() as tape: # Gradient descent
                _, q_values = self.choose_action(state_as_list)
                for i in range(len(q_values)):
                    loss += loss_function([q_values[i][0][agent_instructions[i]]], [updated_q_values[i]])
            loss_list.append(loss)
        loss_average = tf.math.reduce_mean(loss_list).numpy()
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_average

    def prepare_batch(self, replay, batch_size):
        batch = np.asarray(replay)
        if len(replay) > batch_size:
            indices = np.random.randint(0, len(replay), size=batch_size)
            batch = batch[indices]
        return batch


#### Model parameters & creating objects from previous classes


discrete_ordering_pace = 100
reward_div = 10000


## Products

tree_dictionnary = {
    "storage_price": 1,
    "product_name": "tree",
}
tree = Product(tree_dictionnary)

planck_dictionnary = {
    "storage_price": 1,
    "no_sell_penalty": 20,
    "product_name": "planck",
    "sell_price": 10
}
planck = Product(planck_dictionnary)

residues_dictionnary = {
    "storage_price": 1,
    "product_name": "residues",
}
residues = Product(residues_dictionnary)

paper_dictionnary = {
    "storage_price": 1,
    "no_sell_penalty": 20,
    "product_name": "paper",
    "sell_price": 10
}
paper = Product(paper_dictionnary)


## Unlimited raw materials

raw_material_dictionnary = {
    "stock_out_ini": {
        "tree": 10**99,
    },
    "products_stock_out": [tree],
}

raw_material = Unlimitied_stock(raw_material_dictionnary)


## Processes

process_sawing_dictionnary = {
    "stock_in_ini": 0,
    "product_stock_in": tree,
    "product_stock_in_name": "tree",
    "stock_out_ini": {"planck": 0, "residues": 0},
    "products_stock_out": [planck, residues],
    "source": raw_material,
    "delivery_parameters": 1,
    "failure_prob": 0.20,
    "prod_lim": 100,
    "default_rate": 0,
    "process_name": "process_sawing",
    "cost_of_stock_multiplyer": 0.9,
}
process_sawing = Process(process_sawing_dictionnary)

process_paper_maker_dictionnary = {
    "stock_in_ini": 0,
    "product_stock_in": residues,
    "product_stock_in_name": "residues",
    "stock_out_ini": {
        "paper": 0,
    },
    "products_stock_out": [paper],
    "source": process_sawing,
    "delivery_parameters": 1,
    "failure_prob": 0.20,
    "prod_lim": 100,
    "default_rate": 0,
    "process_name": "process_paper_maker",
    "cost_of_stock_multiplyer": 1.1,
}
process_paper_maker = Process(process_paper_maker_dictionnary)

process_list = [process_sawing, process_paper_maker]


## Stores

store_plank_dictionnary = {
    "stock_in_ini": 0,
    "source": process_sawing,
    "product_stock_in": planck,
    "product_stock_in_name": "planck",
    "delivery_parameters": 1,
    "store_name": "store_plank",
    "cost_of_stock_multiplyer": 1.25,
}
store_plank = Store(store_plank_dictionnary)

store_paper_dictionnary = {
    "stock_in_ini": 0,
    "source": process_paper_maker,
    "product_stock_in": paper,
    "product_stock_in_name": "paper",
    "delivery_parameters": 1,
    "store_name": "store_paper",
    "cost_of_stock_multiplyer": 1.25,
}
store_paper = Store(store_paper_dictionnary)

store_list = [store_plank, store_paper]


## Agent_stupid

my_agent_stupid = Agent_stupid()


## Agent_basic

my_agent_basic = Agent_basic()


## Agent_NN

my_agent_NN_dictionnary = {
    "epsilon": 1,
    "decay_rate": 0.99995,
    "discount_rate": 0.6,
}
my_agent_NN = Agent_NN(my_agent_NN_dictionnary)


## Environment

my_environment_dictionnary = {
    "demand_average": 80,
    "first_instructions": [0, 0, 0, 0, 0, 0, 0, 3],
}
my_environment = Environment(my_environment_dictionnary)


## Neural network

state_tf = tf.convert_to_tensor([my_environment.state_as_list], dtype=tf.float32)

input_layer = keras.Input(shape=(len(state_tf[0]),))
first_layer = layers.Dense(
    32,
    activation="relu",
    name="first_layer",
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
)(input_layer)
first_layer_bis = layers.Dense(
    32,
    activation="relu",
    name="first_layer_bis",
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
)(first_layer)
second_layer = layers.Dense(
    32,
    activation="relu",
    name="second_layer",
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
)(first_layer_bis)

process_and_store_intermediary_layers = []
process_and_store_intermediary_layers_bis = []
process_and_store_output_layers = []
for i in range(len(process_list + store_list)):
    process_and_store_intermediary_layers.append(
        layers.Dense(
            24,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),

        )(second_layer)
    )
    process_and_store_intermediary_layers_bis.append(
        layers.Dense(
            24,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),

        )(process_and_store_intermediary_layers[i])
    )
    process_and_store_output_layers.append(
        layers.Dense(
            10,
            activation="linear",
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),

        )(process_and_store_intermediary_layers_bis[i])
    )

model = keras.Model(
    inputs=input_layer,
    outputs=[*process_and_store_output_layers],
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

loss_function = tf.keras.losses.MeanSquaredError()


#### Tensorboard printing


def tensorboard_print(loss=False):
    agent_avg = np.average(agent_history, axis=0)
    state_avg = np.average(state_history, axis=0)
    list_of_output_names = ["z_sawing", "z_paper", "z_store_planck", "z_store_paper"]
    list_of_state_names = ["zz_sawing_in", "zz_sawing_out_plancks", "zz_sawing_out_residues", "zz_paper_in", "zz_paper_out", "zz_store_planck", "zz_store_paper"]

    with summary_writer.as_default():
        if loss:
            tf.summary.scalar("loss", np.average(episode_loss), step=episode)
        tf.summary.scalar(
            "holding_cost",
            np.average(my_environment.holding_cost_history),
            step=episode,
        )
        tf.summary.scalar(
            "missed_sales_penalty",
            np.average(my_environment.missed_sales_penalty_history),
            step=episode,
        )
        tf.summary.scalar(
            "reward", np.average(my_environment.reward_history), step=episode
        )
        tf.summary.scalar(
            "missed_sales", np.average(my_environment.missed_sales_history), step=episode
        )
        tf.summary.scalar(
            "sales", np.average(my_environment.sales_history), step=episode
        )
        tf.summary.scalar(
            "money_revenue", np.average(my_environment.money_revenue_history), step=episode
        )
        for i in range(len(list_of_output_names)):
            tf.summary.scalar(
            list_of_output_names[i],
            agent_avg[i],
            step=episode,
        )
        for i in range(len(list_of_state_names)):
            tf.summary.scalar(
            list_of_state_names[i],
            state_avg[i],
            step=episode,
        )

# tensorboard --logdir logs_2  --port 6066


#### Agent_NN training 
# Different agents cannot be trained at the same time because they all use the same environment. Use "if True / False"


if True:  

    NAME = "model_5_train"
    log_dir = "logs_2/" + NAME
    summary_writer = tf.summary.create_file_writer(log_dir)

    episodes = 2000
    episode_length = 25
    iteractions_between_trains = 5  # Neural network does not train at every iteration
    batch_size = 8
    max_replay = 100

    replay = []
    for episode in range(episodes):
        my_environment.reset()
        episode_loss = []
        agent_history = []
        state_history = []
        for epoch in range(episode_length):
            state = my_environment.state_as_list
            actions, _ = my_agent_NN.choose_action(state)
            (
                state_as_list,
                agent_instructions,
                reward,
                new_state_as_list,
            ) = my_environment.step(actions)    # One iteration is simulated
            replay.append(
                [state_as_list, agent_instructions, reward, new_state_as_list]  # Keeping track for mini-batching
            )
            if len(replay) > max_replay:
                replay = replay[len(replay) - max_replay :]
            if epoch % iteractions_between_trains == 0:
                batch = my_agent_NN.prepare_batch(replay, batch_size)
                loss_average = my_agent_NN.training(batch)
                episode_loss.append(loss_average)
            agent_history.append(agent_instructions)
            state_history.append(my_environment.state_as_list_int)

        tensorboard_print(loss=True)
            
        print("episode is done: ", episode + 1)
        print("epsilon", my_agent_NN.epsilon)

        if episode % 500 == 499:
            model.save("new_models/model_5")


#### Agent_NN testing 
# Different agents cannot be trained at the same time because they all use the same environment. Use "if True / False"


if False:

    model = keras.models.load_model("new_models/model_4")   # Loading previously trained model
    my_agent_NN.epsilon = 0

    NAME = "model_4_test_on_4"
    log_dir = "logs_2/" + NAME
    summary_writer = tf.summary.create_file_writer(log_dir)

    episodes = 500
    episode_length = 25

    for episode in range(episodes):
        my_environment.reset()
        agent_history = []
        state_history = []
        for epoch in range(episode_length):
            state = my_environment.state_as_list
            actions, _ = my_agent_NN.choose_action(state)
            (
                state_as_list,
                agent_instructions,
                reward,
                new_state_as_list,
            ) = my_environment.step(actions)
            agent_history.append(agent_instructions)
            state_history.append(my_environment.state_as_list_int)

        tensorboard_print(loss=False)


#### Agent_basic 
# Different agents cannot be trained at the same time because they all use the same environment. Use "if True / False"


if False:

    NAME = "basic_4"
    log_dir = "logs_2/" + NAME
    summary_writer = tf.summary.create_file_writer(log_dir)

    episodes = 500
    episode_length = 25

    for episode in range(episodes):
        my_environment.reset()
        state_history = []
        agent_history = []
        for epoch in range(episode_length):
            basic_instructions = my_agent_basic.choose_action()
            _, _, basic_reward, _ = my_environment.step(basic_instructions)
            state_history.append(my_environment.state_as_list_int)
            agent_history.append(basic_instructions)

        tensorboard_print(loss=False)


#### Agent_stupid
# Different agents cannot be trained at the same time because they all use the same environment. Use "if True / False"


if False:

    NAME = "stupid_4"
    log_dir = "logs_2/" + NAME
    summary_writer = tf.summary.create_file_writer(log_dir)

    episodes = 500
    episode_length = 25

    for episode in range(episodes):
        my_environment.reset()
        state_history = []
        agent_history = []
        for epoch in range(episode_length):
            stupid_instructions = my_agent_stupid.choose_action()
            _, _, stupid_reward, _ = my_environment.step(stupid_instructions)
            state_history.append(my_environment.state_as_list_int)
            agent_history.append(stupid_instructions)
        
        tensorboard_print(loss=False)

