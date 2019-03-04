riometer_iv.py: riometer_iv.grc
	grcc -d . riometer_iv.grc

clean:
	rm -f riometer_iv.py
	rm -f *.pyc
	rm -f *.csv
	rm -f foobly
	rm -f *.log
	rm -f stop_riometer

install: riometer_iv.py
	cp riometer_iv.py /usr/local/bin
	chmod 755 /usr/local/bin/riometer_iv.py
	cp riometer_helper.py /usr/local/bin
	cp relay_server.py /usr/local/bin
	cp start_riometer $(HOME)
	cp plot_current_data $(HOME)

commit: clean
	git commit -a
	git push
