import threading
from model import *
from screen_capture import start_screen_capture

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    state_dim = (6, 400)
    action_dim = 4

    actor = Actor(input_dim=state_dim, action_dim=action_dim).to(device)
    critic = Critic(input_dim=state_dim).to(device)

    optimizer_actor = optim.Adam(actor.parameters(), lr=0.001)
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.001)

    replay_buffer = ReplayBuffer(max_size=10000)

    # Create a lock for synchronization
    lock = threading.Lock()
    lock.acquire()  # Acquire the lock before starting screen capture

    # Start the screen capture in a separate thread
    threading.Thread(target=start_screen_capture, args=(lock,)).start()

    # Wait for the lock to be released before starting training
    lock.acquire()

    # Train the model
    train(actor, critic, optimizer_actor, optimizer_critic, replay_buffer, num_episodes=1000)

    # Save model parameters
    save_model(actor, critic)

    # Load model parameters (if needed)
    # load_model(actor, critic)
