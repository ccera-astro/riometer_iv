riometer_iv.py: riometer_iv.grc
	grcc -d . riometer_iv.grc

clean:
	rm -f riometer_iv.py
	rm -f *.pyc
	rm -f *.csv

install: riometer_iv.py
	cp riometer_iv.py /usr/local/bin
	cp riometer_helper.py /usr/local/bin
