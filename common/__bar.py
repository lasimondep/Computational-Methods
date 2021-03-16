import progressbar as pb

class Bar(pb.ProgressBar):
    def __init__(self, maxval, **kwargs):
        super().__init__(maxval=maxval, widgets=[
                pb.Percentage(),
                ' (', pb.Counter(), '/%d) ' % maxval,
                pb.Bar(), pb.ETA(),
            ], **kwargs)
