clean:
	find . -type d -name __pycache__ -exec rm -rf {} \+
	find . -type d -name *.pyc -exec rm -rf {} \+ 

ak: clean
	rsync -av ./ --exclude="data" --exclude="outputs" --exclude="lavin.egg-info" ak:/home/quang/workspace/lavin-original/

ri: clean
	rsync -av ./ --exclude="data" --exclude="outputs" --exclude="lavin.egg-info" ri:/home/quang/workspace/lavin-original/

out: clean
	rsync -av ri:/home/quang/workspace/lavin-original/outputs --exclude="data" --exclude="lavin.egg-info" --exclude="*.pth" ./
