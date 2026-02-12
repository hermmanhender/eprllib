"""
Graph plots
============

This module contain all the methods to generate plots to postprocess the results of training.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Union
import numpy as np

# Se asume que estas funciones (get_meter_name, get_variable_name)
# serían métodos de la clase Environment o funciones auxiliares
# que pueden usar env_config para mapear nombres.
# Para este ejemplo, asumiremos que las claves en el diccionario 'info'
# ya son los nombres de las variables de observación.
# Si necesitas un mapeo real, deberías implementar estas funciones
# o pasar un diccionario de mapeo a la clase.
# from eprllib.Environment.Environment import Environment # No se instancia directamente aquí
# def get_meter_name(env_config: Dict[str, Any], meter_id: str) -> str:
#     """Función de ejemplo para obtener el nombre de un medidor."""
#     # Implementación real dependería de la estructura de env_config
#     return f"meter_{meter_id}"

# def get_variable_name(env_config: Dict[str, Any], var_id: str) -> str:
#     """Función de ejemplo para obtener el nombre de una variable de observación."""
#     # Implementación real dependería de la estructura de env_config
#     return f"var_{var_id}"


class ExperienceAnalyzer:
    """
    Clase para analizar los datos de experiencia generados por una simulación
    de agentes en un entorno. Permite extraer estadísticas y generar gráficas
    para evaluar el comportamiento y las estrategias de los agentes.
    """

    def __init__(self, experience_data: Dict[str, Any], env_config: Dict[str, Any]):
        """
        Inicializa el analizador de experiencia.

        Args:
            experience_data (Dict[str, Any]): El diccionario de experiencia
                                               generado por la función
                                               'generate_experience'.
            env_config (Dict[str, Any]): El diccionario de configuración del entorno.
                                         Se incluye según la solicitud del usuario,
                                         aunque su uso directo para mapear nombres
                                         de variables de 'info' dependerá de su estructura.
        """
        if 'experiment' not in experience_data:
            raise ValueError("El diccionario de experiencia debe contener la clave 'experiment'.")

        self.experience_data = experience_data['experiment']
        self.env_config = env_config
        self.all_agents = self._get_all_agents()
        self.num_episodes = len(self.experience_data)

        # Pre-procesar los datos para facilitar el acceso
        self._flat_data = self._flatten_experience_data()

    def _get_all_agents(self) -> List[str]:
        """
        Obtiene una lista de todos los agentes presentes en los datos de experiencia.
        """
        agents = set()
        # Tomar el primer episodio y el primer timestep para obtener los agentes
        first_episode_key = next(iter(self.experience_data))
        first_timestep_key = next(iter(self.experience_data[first_episode_key]))
        
        for agent_id in self.experience_data[first_episode_key][first_timestep_key].keys():
            agents.add(agent_id)
        return sorted(list(agents))

    def _flatten_experience_data(self) -> pd.DataFrame:
        """
        Transforma el diccionario de experiencia anidado en un DataFrame plano
        para facilitar el análisis.
        """
        records = []
        for episode_key, episode_data in self.experience_data.items():
            episode_num = int(episode_key.split('_')[1])
            for timestep_key, timestep_data in episode_data.items():
                timestep_num = int(timestep_key.split('_')[1])
                for agent_id, agent_data in timestep_data.items():
                    record = {
                        'episode': episode_num,
                        'timestep': timestep_num,
                        'agent': agent_id,
                        'observation': agent_data.get('observation'),
                        'action': agent_data.get('action'),
                        'reward': agent_data.get('reward'),
                        'done': agent_data.get('done'),
                        'truncated': agent_data.get('truncated'),
                        'info': agent_data.get('info'),
                        'state': agent_data.get('state') # Incluir estado RNN si existe
                    }
                    records.append(record)
        return pd.DataFrame(records)

    def get_episode_lengths(self, agent_id: str = None) -> Dict[str, int]:
        """
        Calcula la duración (número de timesteps) de cada episodio.

        Args:
            agent_id (str, optional): Si se especifica, calcula la duración
                                      solo para los timesteps donde este agente
                                      estuvo activo. Por defecto, usa el máximo
                                      timestep de cada episodio.

        Returns:
            Dict[str, int]: Un diccionario donde las claves son los números de episodio
                            y los valores son las duraciones.
        """
        episode_lengths = {}
        for episode_key, episode_data in self.experience_data.items():
            episode_num = int(episode_key.split('_')[1])
            if agent_id:
                # Filtrar por agente y obtener el máximo timestep
                filtered_timesteps = [
                    int(ts_key.split('_')[1])
                    for ts_key, ts_data in episode_data.items()
                    if agent_id in ts_data
                ]
                if filtered_timesteps:
                    episode_lengths[f'episode_{episode_num}'] = max(filtered_timesteps)
                else:
                    episode_lengths[f'episode_{episode_num}'] = 0 # Agente no activo en este episodio
            else:
                episode_lengths[f'episode_{episode_num}'] = len(episode_data)
        return episode_lengths

    def get_total_rewards_per_episode(self, agent_id: str = None) -> Dict[str, float]:
        """
        Calcula la recompensa total por episodio para todos los agentes o uno específico.

        Args:
            agent_id (str, optional): El ID del agente para el cual calcular
                                      la recompensa total. Si es None, se calcula
                                      para todos los agentes y se devuelve un diccionario
                                      anidado {agente: {episodio: recompensa_total}}.

        Returns:
            Dict[str, float] o Dict[str, Dict[str, float]]: Recompensa total por episodio.
        """
        if agent_id:
            rewards = {}
            for episode_key, episode_data in self.experience_data.items():
                episode_num = int(episode_key.split('_')[1])
                total_reward = 0.0
                for timestep_data in episode_data.values():
                    if agent_id in timestep_data and timestep_data[agent_id].get('reward') is not None:
                        total_reward += timestep_data[agent_id]['reward']
                rewards[f'episode_{episode_num}'] = total_reward
            return rewards
        else:
            all_agents_rewards = {agent: {} for agent in self.all_agents}
            for agent in self.all_agents:
                all_agents_rewards[agent] = self.get_total_rewards_per_episode(agent)
            return all_agents_rewards

    def get_average_reward_per_timestep(self, agent_id: str = None) -> Dict[str, float]:
        """
        Calcula la recompensa promedio por timestep para todos los agentes o uno específico.

        Args:
            agent_id (str, optional): El ID del agente. Si es None, se calcula
                                      para todos los agentes.

        Returns:
            Dict[str, float]: Recompensa promedio por timestep.
        """
        if agent_id:
            agent_data = self._flat_data[self._flat_data['agent'] == agent_id]
            if not agent_data.empty and 'reward' in agent_data.columns:
                return agent_data['reward'].mean()
            return 0.0
        else:
            avg_rewards = {}
            for agent in self.all_agents:
                avg_rewards[agent] = self.get_average_reward_per_timestep(agent)
            return avg_rewards

    def get_observation_stats(self, agent_id: str, obs_variable_name: str) -> Dict[str, Any]:
        """
        Calcula estadísticas (media, desviación estándar, min, max) para una
        variable de observación específica de un agente.

        Args:
            agent_id (str): El ID del agente.
            obs_variable_name (str): El nombre de la variable de observación
                                     (clave dentro del diccionario 'info').

        Returns:
            Dict[str, Any]: Un diccionario con las estadísticas.
        """
        observations = []
        for _, row in self._flat_data[self._flat_data['agent'] == agent_id].iterrows():
            info = row['info']
            if isinstance(info, dict) and obs_variable_name in info:
                value = info[obs_variable_name]
                # Solo agregar valores escalares (int, float, np.number)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    observations.append(value)
                # Si es array/lista de un solo elemento, tomar ese elemento
                elif isinstance(value, (list, np.ndarray)) and np.array(value).size == 1:
                    observations.append(float(np.array(value).item()))
                # Si es array/lista de más de un elemento, ignorar (no escalar)
                # Si quieres manejar arrays, puedes adaptar aquí

        if not observations:
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "count": 0}

        obs_array = np.array(observations, dtype=float)
        return {
            "mean": np.mean(obs_array),
            "std": np.std(obs_array),
            "min": np.min(obs_array),
            "max": np.max(obs_array),
            "count": len(observations)
        }

    def get_action_distribution(self, agent_id: str) -> Dict[Any, float]:
        """
        Calcula la distribución de las acciones tomadas por un agente.
        Maneja acciones discretas (enteros) y multi-discretas (listas/NDArrays).

        Args:
            agent_id (str): El ID del agente.

        Returns:
            Dict[Any, float]: Un diccionario con la frecuencia de cada acción.
                              Para acciones multi-discretas, las listas/NDArrays
                              se convertirán a tuplas para que sean hashable.
        """
        actions = []
        for _, row in self._flat_data[self._flat_data['agent'] == agent_id].iterrows():
            action = row['action']
            if action is not None:
                if isinstance(action, (list, np.ndarray)):
                    # Convertir a tupla para que sea hashable
                    actions.append(tuple(action))
                else:
                    actions.append(action)
        
        if not actions:
            print(f"No hay acciones registradas para el agente {agent_id}.")
            return {}

        action_series = pd.Series(actions)
        return action_series.value_counts(normalize=True).to_dict()

    def plot_rewards_per_episode(self, agent_id: str = None, figsize: tuple = (10, 6)):
        """
        Genera un gráfico de las recompensas totales por episodio.

        Args:
            agent_id (str, optional): El ID del agente para graficar. Si es None,
                                      grafica las recompensas para todos los agentes.
            figsize (tuple): Tamaño de la figura (ancho, alto).
        """
        plt.figure(figsize=figsize)
        if agent_id:
            rewards = self.get_total_rewards_per_episode(agent_id)
            episodes = list(rewards.keys())
            values = list(rewards.values())
            sns.lineplot(x=episodes, y=values)
            plt.title(f'Recompensa Total por Episodio para el Agente: {agent_id}')
            plt.xlabel('Episodio')
            plt.ylabel('Recompensa Total')
        else:
            all_rewards = self.get_total_rewards_per_episode(None)
            df_plot = pd.DataFrame(all_rewards).reset_index().melt(id_vars='index', var_name='Agent', value_name='Total Reward')
            df_plot.rename(columns={'index': 'Episode'}, inplace=True)
            sns.lineplot(data=df_plot, x='Episode', y='Total Reward', hue='Agent')
            plt.title('Recompensa Total por Episodio para Todos los Agentes')
            plt.xlabel('Episodio')
            plt.ylabel('Recompensa Total')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_action_distribution(self, agent_id: str, figsize: tuple = (10, 6)):
        """
        Genera un gráfico de barras de la distribución de acciones para un agente.

        Args:
            agent_id (str): El ID del agente.
            figsize (tuple): Tamaño de la figura (ancho, alto).
        """
        action_dist = self.get_action_distribution(agent_id)
        if not action_dist:
            print(f"No hay datos de acción para el agente {agent_id} para graficar.")
            return

        actions = [str(a) for a in action_dist.keys()] # Convertir a string para el eje X
        frequencies = list(action_dist.values())

        plt.figure(figsize=figsize)
        sns.barplot(x=actions, y=frequencies)
        plt.title(f'Distribución de Acciones para el Agente: {agent_id}')
        plt.xlabel('Acción')
        plt.ylabel('Frecuencia Relativa')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_observation_variable_over_time(self, agent_id: str, obs_variable_name: str, figsize: tuple = (12, 7)):
        """
        Genera un gráfico de línea de una variable de observación específica
        a lo largo del tiempo para un agente, mostrando cada episodio.

        Args:
            agent_id (str): El ID del agente.
            obs_variable_name (str): El nombre de la variable de observación.
            figsize (tuple): Tamaño de la figura (ancho, alto).
        """
        plot_data = []
        for _, row in self._flat_data[self._flat_data['agent'] == agent_id].iterrows():
            info = row['info']
            if isinstance(info, dict) and obs_variable_name in info:
                value = info[obs_variable_name]
                # Solo agregar valores escalares (int, float, np.number)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    plot_data.append({
                        'episode': row['episode'],
                        'timestep': row['timestep'],
                        'value': float(value)
                    })
                # Si es array/lista de un solo elemento, tomar ese elemento
                elif isinstance(value, (list, np.ndarray)) and np.array(value).size == 1:
                    plot_data.append({
                        'episode': row['episode'],
                        'timestep': row['timestep'],
                        'value': float(np.array(value).item())
                    })
                # Si es array/lista de más de un elemento, ignorar (no escalar)
                # Si quieres manejar arrays, puedes adaptar aquí

        if not plot_data:
            print(f"No hay datos para la variable de observación '{obs_variable_name}' del agente {agent_id}.")
            return

        df_plot = pd.DataFrame(plot_data)
        # Asegurarse de que 'episode' sea tipo str o int, pero no lista/objeto
        df_plot['episode'] = df_plot['episode'].astype(str)

        plt.figure(figsize=figsize)
        sns.lineplot(data=df_plot, x='timestep', y='value', hue='episode', palette='viridis')
        plt.title(f'Variable de Observación \"{obs_variable_name}\" para el Agente: {agent_id} a lo largo del tiempo')
        plt.xlabel('Paso de Tiempo (Timestep)')
        plt.ylabel(obs_variable_name)
        plt.grid(True)
        plt.legend(title='Episodio')
        plt.tight_layout()
        plt.show()

    def plot_cumulative_rewards(self, agent_id: str = None, figsize: tuple = (10, 6)):
        """
        Genera un gráfico de las recompensas acumuladas a lo largo del tiempo.

        Args:
            agent_id (str, optional): El ID del agente para graficar. Si es None,
                                      grafica las recompensas acumuladas para todos los agentes.
            figsize (tuple): Tamaño de la figura (ancho, alto).
        """
        plt.figure(figsize=figsize)
        
        if agent_id:
            df_agent = self._flat_data[self._flat_data['agent'] == agent_id].copy()
            df_agent['cumulative_reward'] = df_agent.groupby('episode')['reward'].cumsum()
            sns.lineplot(data=df_agent, x='timestep', y='cumulative_reward', hue='episode', palette='viridis')
            plt.title(f'Recompensa Acumulada por Episodio para el Agente: {agent_id}')
            plt.xlabel('Paso de Tiempo (Timestep)')
            plt.ylabel('Recompensa Acumulada')
        else:
            df_all = self._flat_data.copy()
            df_all['cumulative_reward'] = df_all.groupby(['agent', 'episode'])['reward'].cumsum()
            sns.lineplot(data=df_all, x='timestep', y='cumulative_reward', hue='agent', style='episode', palette='tab10')
            plt.title('Recompensa Acumulada por Episodio para Todos los Agentes')
            plt.xlabel('Paso de Tiempo (Timestep)')
            plt.ylabel('Recompensa Acumulada')
            plt.legend(title='Agente / Episodio', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_multi_discrete_action_distributions(self, agent_id: str, figsize_per_component: tuple = (8, 5)):
        """
        Genera gráficos de distribución para cada componente de una acción MultiDiscrete
        tomada por un agente.

        Args:
            agent_id (str): El ID del agente.
            figsize_per_component (tuple): Tamaño de la figura para cada gráfico individual (ancho, alto).
        """
        agent_actions = []
        for _, row in self._flat_data[self._flat_data['agent'] == agent_id].iterrows():
            action = row['action']
            if action is not None:
                if isinstance(action, (list, np.ndarray)):
                    agent_actions.append(np.array(action))
                else:
                    # Si no es MultiDiscrete, se lo notifica al usuario y se sale.
                    print(f"Las acciones para el agente {agent_id} no parecen ser MultiDiscrete (listas o np.ndarray).")
                    print("Utiliza 'plot_action_distribution' para acciones discretas o continuas.")
                    return

        if not agent_actions:
            print(f"No hay acciones MultiDiscrete registradas para el agente {agent_id}.")
            return

        # Convertir la lista de arrays de acciones en un solo array 2D
        # donde cada columna es un componente de la acción.
        action_matrix = np.array(agent_actions)
        num_components = action_matrix.shape[1]

        print(f"Generando gráficos de distribución para cada uno de los {num_components} componentes de acción para el agente {agent_id}.")

        for i in range(num_components):
            component_actions = action_matrix[:, i]
            component_dist = pd.Series(component_actions).value_counts(normalize=True).to_dict()

            if not component_dist:
                print(f"No hay datos para el componente de acción {i} del agente {agent_id}.")
                continue

            actions = [str(a) for a in component_dist.keys()]
            frequencies = list(component_dist.values())

            plt.figure(figsize=figsize_per_component)
            sns.barplot(x=actions, y=frequencies)
            plt.title(f'Distribución de Acciones (Componente {i}) para el Agente: {agent_id}')
            plt.xlabel(f'Valor de Acción (Componente {i})')
            plt.ylabel('Frecuencia Relativa')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    def _prepare_correlation_data(self, agent_id: str, obs_variables: List[str]) -> pd.DataFrame:
        """
        Prepara un DataFrame con las variables de observación y los componentes de acción
        para el cálculo de correlación.

        Args:
            agent_id (str): El ID del agente.
            obs_variables (List[str]): Lista de nombres de variables de observación
                                       a extraer del diccionario 'info'.

        Returns:
            pd.DataFrame: DataFrame con las columnas de observación y acción listas para correlación.
        """
        agent_df = self._flat_data[self._flat_data['agent'] == agent_id].copy()
        
        # Extraer variables de observación del diccionario 'info'
        for obs_var in obs_variables:
            agent_df[obs_var] = agent_df['info'].apply(
                lambda x: x.get(obs_var) if isinstance(x, dict) and obs_var in x else np.nan
            )
            # Asegurarse de que los valores sean escalares y numéricos
            agent_df[obs_var] = agent_df[obs_var].apply(
                lambda val: float(np.array(val).item()) if isinstance(val, (list, np.ndarray)) and np.array(val).size == 1 else val
            )
            agent_df[obs_var] = pd.to_numeric(agent_df[obs_var], errors='coerce')

        # Expandir la columna 'action' si es MultiDiscrete
        # Primero, filtrar filas donde la acción es una lista/ndarray y no es None
        multi_discrete_mask = agent_df['action'].apply(lambda x: isinstance(x, (list, np.ndarray)) and x is not None)
        
        if multi_discrete_mask.any():
            # Solo expandir filas donde la acción es lista/ndarray y tiene la misma longitud
            actions_to_expand = agent_df.loc[multi_discrete_mask, 'action']
            # Determinar la longitud más común (modo) de las acciones
            action_lengths = actions_to_expand.apply(lambda x: len(x) if hasattr(x, '__len__') else 0)
            if not action_lengths.empty:
                most_common_len = action_lengths.mode().iloc[0]
                # Filtrar solo las acciones con la longitud más común
                valid_idx = actions_to_expand[action_lengths == most_common_len].index
                # Expandir solo las filas válidas
                action_components_df = pd.DataFrame(
                    list(actions_to_expand.loc[valid_idx]),
                    index=valid_idx
                )
                action_components_df.columns = [f'action_{i}' for i in range(most_common_len)]
                # Inicializar las columnas en el DataFrame principal con NaN
                for col in action_components_df.columns:
                    agent_df[col] = np.nan
                # Asignar los valores expandidos solo a los índices válidos
                agent_df.loc[valid_idx, action_components_df.columns] = action_components_df
                # Asegurarse de que las columnas de acción sean numéricas
                for col in action_components_df.columns:
                    agent_df[col] = pd.to_numeric(agent_df[col], errors='coerce')
            else:
                print(f"Advertencia: El agente {agent_id} tiene acciones que son listas/arrays vacías. No se expandirán componentes de acción.")
                agent_df['action'] = pd.to_numeric(agent_df['action'], errors='coerce')
        else:
            agent_df['action'] = pd.to_numeric(agent_df['action'], errors='coerce')

        # Seleccionar solo las columnas relevantes para la correlación
        cols_to_correlate = obs_variables + [col for col in agent_df.columns if col.startswith('action_')]
        if 'action' in agent_df.columns and not multi_discrete_mask.any():
            cols_to_correlate.append('action')
        cols_to_correlate = list(set(cols_to_correlate))
        correlation_df = agent_df[cols_to_correlate].dropna()
        if correlation_df.empty:
            print(f"Advertencia: No hay datos completos para calcular la correlación para el agente {agent_id} con las variables de observación especificadas.")
            return pd.DataFrame()
        return correlation_df

    def plot_observation_action_correlation(self, agent_id: str, obs_variables: List[str], figsize: tuple = (12, 10)):
        """
        Genera un mapa de calor de la matriz de correlación entre variables de observación
        y componentes de acción para un agente específico.

        Args:
            agent_id (str): El ID del agente.
            obs_variables (List[str]): Lista de nombres de variables de observación
                                       a incluir en el análisis de correlación.
                                       Estas deben ser claves dentro del diccionario 'info'.
            figsize (tuple): Tamaño de la figura (ancho, alto).
        """
        correlation_df = self._prepare_correlation_data(agent_id, obs_variables)

        if correlation_df.empty:
            print("No se pudo generar el gráfico de correlación debido a la falta de datos.")
            return

        # Calcular la matriz de correlación
        correlation_matrix = correlation_df.corr(method='pearson')

        plt.figure(figsize=figsize)
        sns.heatmap(
            correlation_matrix,
            annot=True,      # Mostrar los valores de correlación en el mapa de calor
            cmap='coolwarm', # Esquema de color
            fmt=".2f",       # Formato de los valores (2 decimales)
            linewidths=.5,   # Líneas entre las celdas
            linecolor='black', # Color de las líneas
            cbar_kws={'label': 'Coeficiente de Correlación de Pearson'} # Etiqueta de la barra de color
        )
        plt.title(f'Matriz de Correlación entre Observaciones y Acciones para el Agente: {agent_id}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
    def get_temporal_sequences(self, agent_id: str, variables: List[str]) -> Dict[str, Dict[str, List[Any]]]:
        """
        Extrae secuencias temporales de variables específicas (observaciones, acciones, recompensas)
        para un agente dado, organizadas por episodio.

        Args:
            agent_id (str): El ID del agente.
            variables (List[str]): Lista de nombres de variables a extraer.
                                   Puede incluir 'action', 'reward', 'done', 'truncated', 'state',
                                   y nombres de claves dentro del diccionario 'info' (ej. 'CO2_concentration').
                                   Si 'action' es MultiDiscrete, sus componentes se extraerán
                                   como 'action_0', 'action_1', etc.

        Returns:
            Dict[str, Dict[str, List[Any]]]: Un diccionario donde la primera clave es el episodio,
                                             la segunda clave es el nombre de la variable,
                                             y el valor es la secuencia de esa variable en el tiempo.
                                             Ej: {'episode_1': {'action_0': [v1, v2, ...], 'reward': [r1, r2, ...]}, ...}
        """
        agent_data = self._flat_data[self._flat_data['agent'] == agent_id]
        if agent_data.empty:
            print(f"No hay datos para el agente {agent_id}.")
            return {}

        temporal_sequences = {}

        for episode_num in agent_data['episode'].unique():
            episode_key = f'episode_{episode_num}'
            temporal_sequences[episode_key] = {}
            episode_df = agent_data[agent_data['episode'] == episode_num].sort_values(by='timestep')

            for var_name in variables:
                if var_name == 'action':
                    # Manejar acciones discretas y MultiDiscrete
                    actions_in_episode = episode_df['action'].tolist()
                    if actions_in_episode and isinstance(actions_in_episode[0], (list, np.ndarray)):
                        # MultiDiscrete: expandir a componentes
                        # Determinar la longitud máxima de los componentes de acción en este episodio
                        max_comp = 0
                        for a in actions_in_episode:
                            if isinstance(a, (list, np.ndarray)):
                                max_comp = max(max_comp, len(a))

                        for i in range(max_comp):
                            comp_name = f'action_{i}'
                            temporal_sequences[episode_key][comp_name] = [
                                a[i] if isinstance(a, (list, np.ndarray)) and i < len(a) else np.nan
                                for a in actions_in_episode
                            ]
                    else:
                        # Discreta o continua: un solo componente 'action'
                        temporal_sequences[episode_key]['action'] = episode_df['action'].tolist()
                elif var_name == 'reward':
                    temporal_sequences[episode_key]['reward'] = episode_df['reward'].tolist()
                elif var_name == 'done':
                    temporal_sequences[episode_key]['done'] = episode_df['done'].tolist()
                elif var_name == 'truncated':
                    temporal_sequences[episode_key]['truncated'] = episode_df['truncated'].tolist()
                elif var_name == 'state':
                    # RNN state, might be complex, store as is
                    temporal_sequences[episode_key]['state'] = episode_df['state'].tolist()
                else:
                    # Variable de observación dentro de 'info'
                    obs_values = []
                    for _, row in episode_df.iterrows():
                        info = row['info']
                        if isinstance(info, dict) and var_name in info:
                            value = info[var_name]
                            if isinstance(value, (list, np.ndarray)) and np.array(value).size == 1:
                                obs_values.append(float(np.array(value).item()))
                            elif isinstance(value, (int, float, np.integer, np.floating)):
                                obs_values.append(float(value))
                            else:
                                obs_values.append(np.nan) # Handle non-scalar or complex info values
                        else:
                            obs_values.append(np.nan)
                    temporal_sequences[episode_key][var_name] = obs_values
        return temporal_sequences

    def plot_temporal_profiles(self, agent_id: str, variable_name: str,
                               avg_over_episodes: bool = False, figsize: tuple = (12, 7)):
        """
        Genera un gráfico de línea para una variable específica a lo largo del tiempo
        para un agente, mostrando cada episodio o el promedio sobre ellos.

        Args:
            agent_id (str): El ID del agente.
            variable_name (str): El nombre de la variable a graficar. Puede ser 'action',
                                 'reward', 'done', 'truncated', 'state', o una clave
                                 dentro del diccionario 'info' (ej. 'CO2_concentration', 'action_0').
            avg_over_episodes (bool, optional): Si es True, grafica el promedio y la desviación
                                                estándar de la variable a lo largo de los episodios.
                                                Si es False, grafica cada episodio individualmente.
                                                Defaults to False.
            figsize (tuple): Tamaño de la figura (ancho, alto).
        """
        agent_data = self._flat_data[self._flat_data['agent'] == agent_id].copy()
        if agent_data.empty:
            print(f"No hay datos para el agente {agent_id}.")
            return

        plot_df = pd.DataFrame()

        if variable_name.startswith('action_') or variable_name == 'action':
            # Handle action components or single discrete action
            if variable_name == 'action': # For single discrete action
                plot_df = agent_data[['episode', 'timestep', 'action']].copy()
                plot_df.rename(columns={'action': 'value'}, inplace=True)
            elif variable_name.startswith('action_'): # For multi-discrete action component
                # Extract the component index
                try:
                    comp_idx = int(variable_name.split('_')[1])
                except (IndexError, ValueError):
                    print(f"Nombre de variable de acción inválido: {variable_name}. Debe ser 'action' o 'action_X'.")
                    return
                
                temp_data = []
                for _, row in agent_data.iterrows():
                    action = row['action']
                    if isinstance(action, (list, np.ndarray)) and comp_idx < len(action):
                        temp_data.append({
                            'episode': row['episode'],
                            'timestep': row['timestep'],
                            'value': float(action[comp_idx])
                        })
                    else:
                        temp_data.append({
                            'episode': row['episode'],
                            'timestep': row['timestep'],
                            'value': np.nan # Or some default for missing component
                        })
                plot_df = pd.DataFrame(temp_data).dropna(subset=['value'])

        elif variable_name in ['reward', 'done', 'truncated']:
            plot_df = agent_data[['episode', 'timestep', variable_name]].copy()
            plot_df.rename(columns={variable_name: 'value'}, inplace=True)
            if variable_name == 'done' or variable_name == 'truncated':
                plot_df['value'] = plot_df['value'].astype(int) # Convert boolean to int for plotting

        elif variable_name == 'state':
            print("Graficar estados RNN directamente no es trivial debido a su complejidad. Considera analizar componentes específicos del estado si es posible.")
            return

        else: # Observation variable from 'info'
            plot_data = []
            for _, row in agent_data.iterrows():
                info = row['info']
                if isinstance(info, dict) and variable_name in info:
                    value = info[variable_name]
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        plot_data.append({
                            'episode': row['episode'],
                            'timestep': row['timestep'],
                            'value': float(value)
                        })
                    elif isinstance(value, (list, np.ndarray)) and np.array(value).size == 1:
                        plot_data.append({
                            'episode': row['episode'],
                            'timestep': row['timestep'],
                            'value': float(np.array(value).item())
                        })
                    else:
                        plot_data.append({
                            'episode': row['episode'],
                            'timestep': row['timestep'],
                            'value': np.nan # Non-scalar or complex info values
                        })
                else:
                    plot_data.append({
                        'episode': row['episode'],
                        'timestep': row['timestep'],
                        'value': np.nan
                    })
            plot_df = pd.DataFrame(plot_data).dropna(subset=['value'])
        
        if plot_df.empty:
            print(f"No hay datos válidos para graficar la variable '{variable_name}' para el agente {agent_id}.")
            return

        plt.figure(figsize=figsize)

        if avg_over_episodes:
            # Calculate mean and std deviation per timestep across episodes
            avg_df = plot_df.groupby('timestep')['value'].agg(['mean', 'std']).reset_index()
            sns.lineplot(data=avg_df, x='timestep', y='mean', label='Promedio')
            plt.fill_between(avg_df['timestep'], avg_df['mean'] - avg_df['std'],
                             avg_df['mean'] + avg_df['std'], color='blue', alpha=0.2, label='Desviación Estándar')
            plt.title(f'Perfil Temporal Promedio de "{variable_name}" para el Agente: {agent_id}')
            plt.legend()
        else:
            # Plot each episode individually
            plot_df['episode'] = plot_df['episode'].astype(str) # Ensure hue works correctly
            sns.lineplot(data=plot_df, x='timestep', y='value', hue='episode', palette='viridis')
            plt.title(f'Perfil Temporal de "{variable_name}" por Episodio para el Agente: {agent_id}')
            plt.legend(title='Episodio')

        plt.xlabel('Paso de Tiempo (Timestep)')
        plt.ylabel(variable_name)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
