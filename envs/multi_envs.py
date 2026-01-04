from envs.lbf_wrapper import LBFWrapper
import numpy as np
from typing import List, Any
import multiprocessing as mp

def make_env_fn(env_id, seed, rank, **env_kwargs):
    """
    Helper function to create an environment generator.

    Args:
        env_id (str): Environment ID.
        seed (int): Base random seed.
        rank (int): Index of the environment (used to distinguish seeds for different environments).
        **env_kwargs: Other arguments passed to the environment.

    Returns:
        function: A parameterless function that returns an initialized environment instance when called.
    """
    def _thunk():
        env = LBFWrapper(env_id, **env_kwargs)
        env.reset(seed=seed + rank)
        return env
    return _thunk

def _worker(parent_conn, child_conn, env_fn):
    """
    Worker function for multiprocessing, responsible for running the environment in a separate process.
    Includes Auto-Reset logic: automatically calls reset when the environment ends (done or truncated).

    Args:
        parent_conn: Parent process pipe endpoint (closed in the child process).
        child_conn: Child process pipe endpoint (used for receiving commands and sending data).
        env_fn (function): Function to create the environment.
    """
    parent_conn.close()
    
    try:
        env = env_fn()
        while True:
            cmd, data = child_conn.recv()
            
            if cmd == 'step':
                next_obs, reward, done, truncated, info = env.step(data)
                
                # Check if finished (LBFWrapper returns a bool array, any True means finished)
                is_done = np.any(done) if isinstance(done, (list, np.ndarray)) else done
                is_truncated = np.any(truncated) if isinstance(truncated, (list, np.ndarray)) else truncated
                
                if is_done or is_truncated:
                    # Save terminal state information (Standard Gym practice)
                    info["final_observation"] = next_obs
                    info["final_info"] = info.copy()
                    
                    # Immediately reset environment
                    reset_obs, reset_info = env.reset()
                    
                    # Replace next_obs returned to main process with initial obs of new episode
                    next_obs = reset_obs
                    # Update info
                    info.update(reset_info)
                    
                    # Note: reward is still the reward of this step (terminal step), done is still True
                    # This allows train.py to correctly record episode return and reset RNN
                
                child_conn.send((next_obs, reward, done, truncated, info))
                
            elif cmd == 'reset':
                obs, info = env.reset()
                child_conn.send((obs, info))
                
            elif cmd == 'close':
                env.close()
                child_conn.close()
                break
                
            elif cmd == 'get_attr':
                attr_name = data
                if hasattr(env, attr_name):
                    child_conn.send(getattr(env, attr_name))
                elif hasattr(env.unwrapped, attr_name):
                    child_conn.send(getattr(env.unwrapped, attr_name))
                else:
                    child_conn.send(None)
            
            elif cmd == 'get_env_info':
                if hasattr(env, 'get_env_info'):
                    child_conn.send(env.get_env_info())
                else:
                    child_conn.send(None)

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
                
    except Exception as e:
        try:
            child_conn.send(('ERROR', e))
        except:
            pass
    finally:
        if 'env' in locals():
            env.close()

class MARLAsyncVectorEnv:
    """
    Custom asynchronous parallel vector environment class (Async Vector Env).
    Supports running multiple environments in parallel using multiprocessing and implements Auto-Reset mechanism.

    Main functions:
    1. Execute step and reset for multiple environments in parallel.
    2. Automatically handle reset when environments end to ensure training continuity.
    3. Aggregate data from all environments (obs, reward, done, etc.) into stacked numpy arrays.
    """
    def __init__(self, env_fns: List[Any]):
        """
        Initialize the parallel environment.

        Args:
            env_fns (list): List of environment creation functions.
        """
        self.num_envs = len(env_fns)
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        
        self.processes = []
        for parent_conn, child_conn, env_fn in zip(self.parent_conns, self.child_conns, env_fns):
            p = mp.Process(target=_worker, args=(parent_conn, child_conn, env_fn))
            p.daemon = True 
            p.start()
            child_conn.close() 
            self.processes.append(p)
            
        self.n_agents = None 

    def reset(self):
        """
        Reset all parallel environments.

        Returns:
            obs_stack (np.ndarray): Stacked initial observations. Shape: (num_envs, n_agents, ...).
            info_list (list): List containing info from all environments.
        """
        for conn in self.parent_conns:
            conn.send(('reset', None))
            
        results = [conn.recv() for conn in self.parent_conns]
        self._check_errors(results)
        
        obs_list, info_list = zip(*results)
        
        if self.n_agents is None:
             self.n_agents = obs_list[0].shape[0]

        return np.stack(obs_list), list(info_list)

    def step(self, actions_list):
        """
        Execute one step of action in all parallel environments.

        Args:
            actions_list (list or np.ndarray): List or array containing actions for each environment.
                                               Shape: (num_envs, n_agents) or (num_envs, n_agents, action_dim).

        Returns:
            next_obss (np.ndarray): Stacked next observations. Shape: (num_envs, n_agents, ...).
            rewards (np.ndarray): Stacked rewards. Shape: (num_envs, n_agents).
            dones (np.ndarray): Stacked done flags. Shape: (num_envs, n_agents).
            truncateds (np.ndarray): Stacked truncated flags. Shape: (num_envs, n_agents).
            infos (list): List containing info from all environments. If environment auto-resets, contains 'final_observation' and 'final_info'.
        """
        if len(actions_list) != self.num_envs:
            raise ValueError(f"Actions length mismatch! Expected {self.num_envs}, got {len(actions_list)}")

        for conn, act in zip(self.parent_conns, actions_list):
            conn.send(('step', act))
            
        results = [conn.recv() for conn in self.parent_conns]
        self._check_errors(results)
        
        next_obss, rewards, dones, truncateds, infos = zip(*results)
        
        return (
            np.stack(next_obss), 
            np.stack(rewards), 
            np.stack(dones), 
            np.stack(truncateds), 
            list(infos)
        )

    def get_env_info(self):
        """
        Get environment information (assuming all environments are the same, only get from the first one).

        Returns:
            info (dict): Environment information dictionary.
        """
        self.parent_conns[0].send(('get_env_info', None))
        result = self.parent_conns[0].recv()
        if isinstance(result, tuple) and result[0] == 'ERROR':
            raise result[1]
        return result

    def _check_errors(self, results):
        for item in results:
            # Add type check to prevent mistaking Numpy array for Error string
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and item[0] == 'ERROR':
                raise item[1]

    def close(self):
        """
        Close all child processes and pipes.
        """
        for conn in self.parent_conns:
            try:
                conn.send(('close', None))
            except:
                pass
        for p in self.processes:
            p.join()