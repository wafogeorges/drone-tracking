import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, OffboardError
async def aller_a_position(drone, nord, est, altitude_cible, tolerance=0.1):
    print(f"Déplacement vers ({nord}, {est})...")
    await drone.offboard.set_position_ned(
        PositionNedYaw(nord, est, -altitude_cible, 0.0)
    )
    async for pos_velocity in drone.telemetry.position_velocity_ned():
        pos = pos_velocity.position
        distance = ((pos.north_m - nord)**2 + (pos.east_m - est)**2) ** 0.5
        print(f"Position actuelle : nord={pos.north_m:.1f} est={pos.east_m:.1f} (distance restante: {distance:.1f} m)")
        if distance < tolerance:
            print("Position atteinte !")
            break

async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("En attente de connexion au drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connecté !")
            break

    print("Armement...")
    await drone.action.arm()

    print("Décollage...")
    await drone.action.takeoff()

    altitude_cible = 5.0
    async for position in drone.telemetry.position():
        altitude_actuelle = position.relative_altitude_m
        print(f"Altitude actuelle : {altitude_actuelle:.1f} m")
        if altitude_actuelle >= altitude_cible * 0.95:
            print("Altitude cible atteinte !")
            break

    print("Envoi d'une première commande de position...")
    await drone.offboard.set_position_ned(
        PositionNedYaw(0.0, 0.0, -altitude_cible, 0.0)
    )

    print("Activation du mode Offboard...")
    try:
        await drone.offboard.start()
    except OffboardError as erreur:
        print(f"Échec de l'activation Offboard : {erreur}")
        await drone.action.land()
        return

    print("Déplacement vers l'avant (5m)...")
    await aller_a_position(drone, 5.0, 0.0, altitude_cible)
    print("Retour à la position de départ...")
    await aller_a_position(drone, 0.0, 0.0, altitude_cible)
    print("Désactivation du mode Offboard...")
    await drone.offboard.stop()
    print("Atterrissage...")
    await drone.action.land()

asyncio.run(run())
