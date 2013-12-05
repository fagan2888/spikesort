""" 
.. module:: catalog
    :synopsis:  Catalog for storing metadata about sorted units.
    
.. moduleauthor:: Mat Leonard <leonard.mat@gmail.com>
"""

import datetime

import sqlalchemy as sql
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Session(Base):
    ''' Class for storing information about recording sessions, including 
        units sorted from the data.

    **Arguments**:
        *rat*: string
         The name of the rat the units came from.
        
        *date*: :class:`datetime.date`
         Date of the session.
    
    **Attributes**:
        *notes*: string 
         Any notes about the session.
    
        *duration*: float
         The session's duration.

        *depth*: float
         Depth of the recording.

        *path*: string
         File path to the session data files.

        *units*: list of :class:`Unit`
         List of units recorded during the session.

    '''
    __tablename__ = 'sessions'
    
    id = sql.Column(sql.Integer, primary_key=True)
    
    rat = sql.Column(sql.String)
    date = sql.Column(sql.Date)
    notes = sql.Column(sql.String)
    duration = sql.Column(sql.Float)
    depth = sql.Column(sql.Float)
    path = sql.Column(sql.String)
    
    units = sql.orm.relationship("Unit", 
                                 order_by="Unit.id", 
                                 backref="session",
                                 cascade="all, delete, delete-orphan")
    
    def __init__(self, rat, date):
        self.rat = rat
        self.date = date
    
    def __repr__(self):
        return "<Session(%s, %s)>" % (self.rat, self.date)

class Unit(Base):
    ''' Class for storing information about sorted units.

    **Arguments**:
        *tetrode*: int
         Tetrode number unit was recorded on.
        
        *cluster*: int
         Cluster number of the unit from the sorting.
        
    **Attributes**:
        *rate*: float
         Average firing rate of the unit.
        
        *falsePositive*: float
         False positive rate, from cluster metrics.
        
        *falseNegative*: float
         False negative rate, from cluster metrics.
        
        *notes*: string
         Any notes about the unit.

    '''
    __tablename__ = 'units'
    
    id = sql.Column(sql.Integer, primary_key=True)

    tetrode = sql.Column(sql.Integer)
    cluster = sql.Column(sql.Integer)
    falsePositive = sql.Column(sql.Float)
    falseNegative = sql.Column(sql.Float)
    notes = sql.Column(sql.String)
    rate = sql.Column(sql.Float)
    
    session_id = sql.Column(sql.Integer, sql.ForeignKey("sessions.id"))
    
    def __init__(self, tetrode, cluster):
        self.tetrode = tetrode
        self.cluster = cluster
    
    def __repr__(self):
        return "<Unit(%s, %s, tetrode %s, cluster %s)>" \
            % (self.id, self.session, self.tetrode, self.cluster)

class Catalog(object):
    """ Catalog for storing session and unit metadata.
        
        **Arguments**:
            *database*: string
             Path to database to load.

        **Attributes**: 
            *Session*: :class:`Session`
             Session class used in creating new sessions and querying sessions.

            *Unit*: :class:`Unit`
             Unit class used in creating new units and querying units.

            *date*: :py:class:`datetime.date`
             Convience date object for use when creating a :class:`Session`.

    """
    def __init__(self, database):
        self.database = database
        self.engine = sql.create_engine('sqlite:///'+database, echo = False)
        self._ConnectGen = sql.orm.sessionmaker(bind=self.engine)
        self.connection = self._ConnectGen(expire_on_commit=False)
        
        Base.metadata.create_all(self.engine)

        self.Session = Session
        self.Unit = Unit
        self.date = datetime.date

    def add(self, object):
        """ Add a :class:`Session` or a :class:`Unit` to the catalog. """
        self.connection.add(object)

    def delete(self, object):
        """ Delete a :class:`Session` or a :class:`Unit` from the catalog. """
        self.connection.delete(object)
        
    def save(self):
        """ Save changes to the catalog database. """
        try:
            self.connection.commit()
        except Exception as e:
            self.undo()
            raise e

    def undo(self):
        """ Undo last changes to the catalog. """
        self.connection.rollback()

    def query(self, *args, **kwargs):
        """ Queries the catalog.  This uses the `SQLAlchemy querying API 
            <http://docs.sqlalchemy.org/en/rel_0_8/orm/query.html>`_. """
        return self.connection.query(*args, **kwargs)
    
    def __getitem__(self, unit_ids):
        
        try:
            return self.query(Unit).filter(Unit.id == unit_ids).one()
        except sql.exc.InterfaceError:
            return self.query(Unit).filter(Unit.id.in_(unit_ids)).all()

    def __repr__(self):
        return "Unit Catalog stored at {}".format(self.database)



def open(database=':memory:'):
    """ Return a Catalog object using the given database. """
    return Catalog('/'+database)
