Catalog
*******

Typical usage
-------------
The Catalog class makes a connection to database containing metadata about sorted units.  After using :mod:`process` and :mod:`cluster`, you will want to store information about the units found.

First you'll want to connect to a database. ::

    from spikesort import catalog
    database = '/path/to/database.sql'
    log = catalog.open(database)

To add a session, you can do ::

    session = log.Session('RATNAME', log.date(13, 9, 4))
    session.path = '/path/to/datafiles'
    log.add(session)
    log.save()

After sorting the data you hopefully found some units, so let's add them to the session. ::

    tetrode = 1
    units = [ log.Unit(tetrode, cluster_id) for cluster_id in clusters ]
    session.units = units
    log.save()

When you want to retrieve units or sessions from you can do something like: ::

    sessions = log.query(log.Session).all()

This will return all sessions.  You can also select by ratname or date, ::

    session = log.query(log.Session).filter(log.Session.rat == 'RATNAME').all()
    session = log.query(log.Session).filter(log.Session.date < log.date(13, 9, 4)).all()
    units = log.query(log.Unit).filter(log.Unit.rate > 0.5).all()

The catalog is built with `SQLAlchemy <http://www.sqlalchemy.org/>`_.  For an introduction to querying, see `SQLAlchemy Object Relational Introduction <http://docs.sqlalchemy.org/en/rel_0_8/orm/tutorial.html#querying>`_.  More documentation can be found at the `Querying API documentation <http://docs.sqlalchemy.org/en/rel_0_8/orm/query.html>`_.

Functions
---------
.. automodule:: catalog
    :members: open

Classes
-------
.. automodule:: catalog
    :members: Catalog, Session, Unit