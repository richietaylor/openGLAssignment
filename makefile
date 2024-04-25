install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt
	. venv/bin/activate; pip3 install -Ur post_requirements.txt

venv :
	test -d venv || python3 -m venv venv

clean:
	rm -rf venv