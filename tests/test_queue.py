from src.simulation.queue_manager import StationQueue
def test_release_and_promote():
    station=StationQueue(1); assert station.arrive(5)=="charging"; assert station.arrive(10)=="queued"
    result=station.advance(5)
    assert result=={"completed":1,"promoted":1} and station.occupied_chargers==1 and station.queue_length==0 and station.free_chargers==0
    station.advance(10); assert station.occupied_chargers==0 and station.free_chargers==1
def test_wait_uses_sessions_and_queue():
    station=StationQueue(2); station.arrive(10); station.arrive(20); station.arrive(15)
    assert station.estimate_wait_minutes()==20
