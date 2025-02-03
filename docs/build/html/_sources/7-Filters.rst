Filters API
============

The observations are the variables, parameters, meters, and others that the agent is 
capable to sense. The following code-block present an example of an observation configuration
with the help of ``ObservationSpec`` to show how is the correct way of set this agent property.

.. code-block:: python

    from eprllib.Agent.AgentSpec import ObservationSpec
    
    observation = ObservationSpec(
        variables = [
            ("Site Outdoor Air Drybulb Temperature", "Environment"),
            ("Site Wind Speed", "Environment"),
            ("Site Outdoor Air Relative Humidity", "Environment"),
            ("Zone Mean Air Temperature", "Thermal Zone: Room1"),
            ("Zone Air Relative Humidity", "Thermal Zone: Room1"),
            ("Zone People Occupant Count", "Thermal Zone: Room1"),
            ("Zone Thermal Comfort CEN 15251 Adaptive Model Category I Status", "Room1 Occupancy"),
            ("Zone Thermal Comfort CEN 15251 Adaptive Model Category II Status", "Room1 Occupancy"),
            ("Zone Thermal Comfort CEN 15251 Adaptive Model Category III Status", "Room1 Occupancy"),
        ],
        internal_variables = [
            ("Zone Air Volume", "Thermal Zone: Room1"),
            ("Zone Floor Area", "Thermal Zone: Room1"),
        ],
        simulation_parameters = {
            'today_weather_horizontal_ir_at_time': True,
        },
        meters = [
            "Electricity:Building",
            "Heating:DistrictHeatingWater",
            "Cooling:DistrictCooling",
        ],
        use_one_day_weather_prediction = True,
        prediction_hours = 3,
        prediction_variables = {
            'outdoor_dry_bulb': True,
        },
        use_actuator_state = True,
        other_obs = {
            "WWR-North": ((2.39-0.01)*(2.28-1.48))/((3.08)*(2.4)), # WWR: Window-Wall Ratio
            "WWR-South": ((2.39-0.01)*(2.28-1.48))/((3.08)*(2.4)),
            "WWR-East": 0.0,
            "WWR-West": 0.0,
        }
    ),


Note that ``observation_fn`` is not part of the observation capabilities of the agent. The observation 
function is defined in the environment because it defines the relationship between agents as well.

